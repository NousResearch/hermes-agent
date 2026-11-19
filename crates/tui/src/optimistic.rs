//! Optimistic UI: paint first when success is likely, reconcile on the
//! gateway, roll back or offer Undo on failure.
//!
//! Vercel Web Interface Guidelines — Optimistic updates.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::rpc::GatewayClient;
use crate::state::{AppState, PermissionMode};

pub const TOAST_SECS: f64 = 3.5;
pub const UNDO_SECS: f64 = 8.0;
pub const UNDO_MAX_BYTES: usize = 2 * 1024 * 1024;
/// Skip a loader that never made a frame (same-drain start+end).
pub const LOAD_SHOW_DELAY: Duration = Duration::from_millis(180);
/// Once a loader is on screen, keep it at least this long.
pub const LOAD_MIN_VISIBLE: Duration = Duration::from_millis(400);

pub fn toast_ttl() -> Duration {
    Duration::from_millis((TOAST_SECS * 1000.0) as u64)
}

pub fn undo_ttl() -> Duration {
    Duration::from_millis((UNDO_SECS * 1000.0) as u64)
}

pub fn brief_err(e: &impl std::fmt::Display) -> String {
    e.to_string()
        .lines()
        .next()
        .unwrap_or("failed")
        .chars()
        .take(56)
        .collect()
}

/// `config.set` yolo returns `"1"` / `"0"` (Ink) and may send a bool.
pub fn yolo_from_value(v: &Value) -> Option<bool> {
    match v.get("value") {
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "on" | "yolo" => Some(true),
            "0" | "false" | "off" | "ask" | "manual" => Some(false),
            _ => None,
        },
        Some(Value::Bool(b)) => Some(*b),
        Some(Value::Number(n)) => n.as_i64().map(|i| i != 0),
        _ => None,
    }
}

pub fn apply_yolo_toggle(s: &mut AppState) -> (PermissionMode, u64) {
    let prev = s.metrics.permission_mode;
    let next = prev.toggle();
    s.metrics.permission_mode = next;
    s.yolo_epoch = s.yolo_epoch.wrapping_add(1);
    s.set_toast(format!("Mode: {}", next.label()));
    (prev, s.yolo_epoch)
}

pub fn apply_mode_cycle(s: &mut AppState) -> (PermissionMode, PermissionMode, u64) {
    let prev = s.metrics.permission_mode;
    let next = prev.cycle();
    s.metrics.permission_mode = next;
    s.yolo_epoch = s.yolo_epoch.wrapping_add(1);
    s.set_toast(format!("Mode: {} · shift+tab cycles", next.label()));
    (prev, next, s.yolo_epoch)
}

pub fn apply_plan_mode(s: &mut AppState) -> (PermissionMode, PermissionMode, u64) {
    let prev = s.metrics.permission_mode;
    let next = PermissionMode::Plan;
    s.metrics.permission_mode = next;
    s.yolo_epoch = s.yolo_epoch.wrapping_add(1);
    s.set_toast("Mode: plan · writes denied until you leave");
    (prev, next, s.yolo_epoch)
}

pub fn spawn_mode_reconcile(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    sid: String,
    epoch: u64,
    prev: PermissionMode,
    next: PermissionMode,
) {
    let yolo_rpc = PermissionMode::needs_yolo_rpc(prev, next);
    let plan_rpc = PermissionMode::needs_plan_rpc(prev, next);
    if !yolo_rpc && !plan_rpc {
        return;
    }
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        if plan_rpc {
            let plan_on = next == PermissionMode::Plan;
            match client
                .set_config(json!({
                    "key": "plan",
                    "session_id": sid,
                    "value": if plan_on { "on" } else { "off" },
                }))
                .await
            {
                Ok(_) => {}
                Err(e) => {
                    let mut s = state.lock().await;
                    if s.yolo_epoch != epoch {
                        return;
                    }
                    s.metrics.permission_mode = prev;
                    s.set_toast(format!(
                        "plan reverted · {} · shift+tab retries",
                        brief_err(&e)
                    ));
                    return;
                }
            }
        }
        if yolo_rpc {
            let yolo_on = next == PermissionMode::Yolo;
            match client
                .set_config(json!({
                    "key": "yolo",
                    "session_id": sid,
                    "value": if yolo_on { "on" } else { "off" },
                }))
                .await
            {
                Ok(v) => {
                    let Some(on) = yolo_from_value(&v) else {
                        return;
                    };
                    let mut s = state.lock().await;
                    if s.yolo_epoch != epoch {
                        return;
                    }
                    if next == PermissionMode::Plan {
                        s.metrics.permission_mode = PermissionMode::Plan;
                    } else {
                        s.metrics.permission_mode = PermissionMode::from_session_info(on);
                    }
                    s.mark_dirty();
                }
                Err(e) => {
                    let mut s = state.lock().await;
                    if s.yolo_epoch != epoch {
                        return;
                    }
                    s.metrics.permission_mode = prev;
                    s.set_toast(format!(
                        "mode reverted · {} · shift+tab retries",
                        brief_err(&e)
                    ));
                }
            }
        } else {
            let mut s = state.lock().await;
            if s.yolo_epoch == epoch {
                s.metrics.permission_mode = next;
                s.mark_dirty();
            }
        }
    });
}

pub fn spawn_plan_deny(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    sid: String,
    request_id: Option<String>,
) {
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        match client
            .approval_respond(&sid, "deny", request_id.as_deref())
            .await
        {
            Ok(_) => {
                let mut s = state.lock().await;
                s.pending_approval = None;
                s.set_toast("plan mode denied a write · shift+tab to ask");
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("plan deny failed · {}", brief_err(&e)));
            }
        }
    });
}

pub fn spawn_interrupt(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>, sid: String) {
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        match client.interrupt(&sid).await {
            Ok(()) => {
                state.lock().await.set_toast("Interrupted");
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("still running · {} · esc retries", brief_err(&e)));
            }
        }
    });
}

pub fn spawn_redirect(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    sid: String,
    text: String,
) {
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        if let Err(e) = client.session_redirect(&sid, &text).await {
            let mut s = state.lock().await;
            s.enqueue(text);
            s.set_toast(format!("redirect failed · {} · queued", brief_err(&e)));
        }
    });
}

pub fn spawn_steer(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    sid: String,
    text: String,
) {
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        if let Err(e) = client.steer(&sid, &text).await {
            let mut s = state.lock().await;
            s.enqueue(text);
            s.set_toast(format!("back in queue · {} · enter sends", brief_err(&e)));
        }
    });
}

pub fn spawn_submit(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    sid: String,
    text: String,
) {
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        if let Err(e) = client.submit_prompt(&sid, &text).await {
            let mut s = state.lock().await;
            s.finish_streaming();
            s.set_toast(format!("not sent · {} · enter retries", brief_err(&e)));
        }
    });
}

pub fn spawn_compress(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>, sid: String) {
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        if let Err(e) = client.compress(&sid).await {
            state.lock().await.set_toast(format!(
                "still full · {} · /compress retries",
                brief_err(&e)
            ));
        }
    });
}

pub fn spawn_model(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    sid: String,
    epoch: u64,
    prev_model: String,
    prev_provider: String,
    value: String,
) {
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        match client
            .set_config(json!({
                "key": "model",
                "value": value,
                "session_id": sid,
            }))
            .await
        {
            Ok(_) => {}
            Err(e) => {
                let mut s = state.lock().await;
                if s.model_epoch != epoch {
                    return;
                }
                s.metrics.active_model = prev_model;
                s.metrics.active_provider = prev_provider;
                s.set_toast(format!(
                    "model reverted · {} · ctrl+o to pick again",
                    brief_err(&e)
                ));
            }
        }
    });
}

pub fn spawn_agents_pause(state: &Arc<Mutex<AppState>>, client: &Arc<GatewayClient>, paused: bool) {
    let state = Arc::clone(state);
    let client = Arc::clone(client);
    tokio::spawn(async move {
        if let Err(e) = client.delegation_pause(paused).await {
            let mut s = state.lock().await;
            s.agents_paused = !paused;
            s.set_toast(format!("pause reverted · {} · p retries", brief_err(&e)));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn yolo_parses_ink_values() {
        assert_eq!(yolo_from_value(&json!({"value": "1"})), Some(true));
        assert_eq!(yolo_from_value(&json!({"value": "0"})), Some(false));
        assert_eq!(yolo_from_value(&json!({"value": true})), Some(true));
        assert_eq!(yolo_from_value(&json!({"value": 0})), Some(false));
        assert_eq!(yolo_from_value(&json!({"value": "on"})), Some(true));
        assert_eq!(yolo_from_value(&json!({})), None);
    }

    #[test]
    fn brief_err_takes_first_line() {
        assert_eq!(brief_err(&"boom\nstack"), "boom");
    }

    #[test]
    fn yolo_toggle_bumps_epoch() {
        let mut s = AppState::new();
        assert_eq!(s.metrics.permission_mode, PermissionMode::Manual);
        let (prev, epoch) = apply_yolo_toggle(&mut s);
        assert_eq!(prev, PermissionMode::Manual);
        assert_eq!(s.metrics.permission_mode, PermissionMode::Yolo);
        assert_eq!(epoch, 1);
        let (prev, epoch) = apply_yolo_toggle(&mut s);
        assert_eq!(prev, PermissionMode::Yolo);
        assert_eq!(s.metrics.permission_mode, PermissionMode::Manual);
        assert_eq!(epoch, 2);
        let (prev, next, epoch) = apply_mode_cycle(&mut s);
        assert_eq!(prev, PermissionMode::Manual);
        assert_eq!(next, PermissionMode::Yolo);
        assert_eq!(epoch, 3);
        let (prev, next, _) = apply_mode_cycle(&mut s);
        assert_eq!(prev, PermissionMode::Yolo);
        assert_eq!(next, PermissionMode::Plan);
        let (prev, next, _) = apply_plan_mode(&mut s);
        assert_eq!(prev, PermissionMode::Plan);
        assert_eq!(next, PermissionMode::Plan);
    }
}
