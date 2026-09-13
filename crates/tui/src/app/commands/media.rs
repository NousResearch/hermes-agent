//! Attach, clipboard, and image command handlers.
use ratatui_textarea::TextArea;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::state::{format_spawn_diff, parse_spawn_entries, resolve_spawn_entry, AppState};

use super::super::paste_lead;

pub(crate) async fn attach_named_file(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
    arg: &str,
) {
    let path = arg.trim();
    if path.is_empty() {
        state.lock().await.set_toast("usage: /file <path>");
        return;
    }
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    match client.file_attach(&sid, path).await {
        Ok(v) if v.get("attached").and_then(|x| x.as_bool()) == Some(true) => {
            let refer = v.get("ref_text").and_then(|s| s.as_str()).unwrap_or(path);
            textarea.insert_str(format!("{}{refer} ", paste_lead(textarea)));
            state.lock().await.set_toast("file attached");
        }
        Ok(v) => {
            let msg = v
                .get("message")
                .and_then(|s| s.as_str())
                .unwrap_or("attach failed");
            state.lock().await.set_toast(msg);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("file · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn attach_named_pdf(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
    arg: &str,
) {
    let path = arg.trim();
    if path.is_empty() {
        state.lock().await.set_toast("usage: /pdf <path>");
        return;
    }
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    match client.pdf_attach(&sid, path).await {
        Ok(v) if v.get("attached").and_then(|x| x.as_bool()) == Some(true) => {
            let n = v
                .get("pages_attached")
                .and_then(|x| x.as_u64())
                .unwrap_or(0);
            if let Some(pages) = v.get("pages").and_then(|x| x.as_array()) {
                for page in pages {
                    if let Some(p) = page.get("path").and_then(|x| x.as_str()) {
                        state
                            .lock()
                            .await
                            .remember_image(std::path::PathBuf::from(p));
                    }
                }
            }
            let note = v
                .get("text")
                .and_then(|s| s.as_str())
                .unwrap_or("PDF attached");
            textarea.insert_str(format!("{}{note} ", paste_lead(textarea)));
            state.lock().await.set_toast(format!("pdf · {n} pages"));
        }
        Ok(v) => {
            let msg = v
                .get("message")
                .and_then(|s| s.as_str())
                .unwrap_or("pdf attach failed");
            state.lock().await.set_toast(msg);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("pdf · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn detach_image(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    _textarea: &mut TextArea<'static>,
    path: &str,
) {
    let (sid, chosen) = {
        let s = state.lock().await;
        let sid = s.session_id.clone();
        let chosen = if path.is_empty() {
            s.pending_images
                .last()
                .map(|p| p.path.display().to_string())
        } else {
            Some(path.to_string())
        };
        (sid, chosen)
    };
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    let Some(chosen) = chosen else {
        state.lock().await.set_toast("no attached image");
        return;
    };
    match client.image_detach(&sid, &chosen).await {
        Ok(v) => {
            let detached = v.get("detached").and_then(|x| x.as_bool()) == Some(true);
            let mut s = state.lock().await;
            s.pending_images
                .retain(|p| p.path.display().to_string() != chosen);
            s.mark_dirty();
            s.set_toast(if detached {
                format!("detached {chosen}")
            } else {
                "not attached on gateway".into()
            });
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("detach · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn imagine_image(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let prompt = arg.trim();
    if prompt.is_empty() {
        match client.image_generate(None, true).await {
            Ok(v) => {
                let available = v.get("available").and_then(|x| x.as_bool()) == Some(true);
                state.lock().await.set_toast(if available {
                    "image gen ready · /imagine <prompt>"
                } else {
                    "no image backend · hermes tools"
                });
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("imagine · {}", optimistic::brief_err(&e)));
            }
        }
        return;
    }
    state.lock().await.set_toast("generating image…");
    match client.image_generate(Some(prompt), false).await {
        Ok(v) => {
            if v.get("success").and_then(|x| x.as_bool()) != Some(true) {
                let err = v
                    .get("error")
                    .and_then(|x| x.as_str())
                    .unwrap_or("generation failed");
                state.lock().await.set_toast(err);
                return;
            }
            let path = v.get("image").and_then(|x| x.as_str()).unwrap_or("");
            let has_data = v.get("image_data").and_then(|x| x.as_str()).is_some();
            let body = if path.is_empty() {
                if has_data {
                    "generated (data URL omitted — attach from the provider path)".into()
                } else {
                    "generated".into()
                }
            } else {
                format!("{path}\n/image {path} to attach")
            };
            state.lock().await.open_peek("imagine".into(), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("imagine · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn replay_diff(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    arg: &str,
) {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    if parts.len() != 2 {
        state
            .lock()
            .await
            .set_toast("usage: /replay-diff <a> <b>  (indexes from /replay)");
        return;
    }
    let sid = state.lock().await.session_id.clone().unwrap_or_default();
    let entries = match client.spawn_tree_list(&sid).await {
        Ok(v) => parse_spawn_entries(&v),
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("replay-diff · {}", optimistic::brief_err(&e)));
            return;
        }
    };
    let Some(a) = resolve_spawn_entry(parts[0], &entries) else {
        state.lock().await.set_toast(format!(
            "replay-diff: could not resolve {} · /replay lists 1..{}",
            parts[0],
            entries.len()
        ));
        return;
    };
    let Some(b) = resolve_spawn_entry(parts[1], &entries) else {
        state.lock().await.set_toast(format!(
            "replay-diff: could not resolve {} · /replay lists 1..{}",
            parts[1],
            entries.len()
        ));
        return;
    };
    let a_path = a.path.clone();
    let b_path = b.path.clone();
    let a_ref = parts[0].to_string();
    let b_ref = parts[1].to_string();
    let loaded = tokio::try_join!(
        client.spawn_tree_load(&a_path),
        client.spawn_tree_load(&b_path)
    );
    match loaded {
        Ok((av, bv)) => {
            let body = format_spawn_diff(&av, &bv, &a_ref, &b_ref);
            state
                .lock()
                .await
                .open_peek("replay-diff".into(), body, None);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("replay-diff · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn attach_named_image(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
    arg: &str,
) {
    let path = arg.trim();
    if path.is_empty() {
        state.lock().await.set_toast("usage: /image <path>");
        return;
    }
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    match client.image_attach(&sid, path).await {
        Ok(v) if v.get("attached").and_then(|x| x.as_bool()) == Some(true) => {
            let stored = v.get("path").and_then(|s| s.as_str()).unwrap_or(path);
            insert_image_token(textarea, stored);
            state
                .lock()
                .await
                .remember_image(std::path::PathBuf::from(stored));
            state.lock().await.set_toast("image attached");
        }
        Ok(v) => {
            let msg = v
                .get("message")
                .and_then(|s| s.as_str())
                .unwrap_or("attach failed");
            state.lock().await.set_toast(msg);
        }
        Err(e) => {
            state
                .lock()
                .await
                .set_toast(format!("image · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn paste_clipboard_image(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) {
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.set_toast("no session");
        return;
    };
    match client.clipboard_paste(&sid).await {
        Ok(v) if v.get("attached").and_then(|x| x.as_bool()) == Some(true) => {
            let path = v.get("path").and_then(|s| s.as_str()).unwrap_or("");
            insert_image_token(textarea, path);
            if !path.is_empty() {
                state
                    .lock()
                    .await
                    .remember_image(std::path::PathBuf::from(path));
            }
            state.lock().await.set_toast("clipboard image attached");
        }
        Ok(v) => {
            if attach_clipboard_bytes(state, client, textarea, &sid).await {
                return;
            }
            let msg = v
                .get("message")
                .and_then(|s| s.as_str())
                .unwrap_or("no clipboard image");
            state.lock().await.set_toast(msg);
        }
        Err(e) => {
            if attach_clipboard_bytes(state, client, textarea, &sid).await {
                return;
            }
            state
                .lock()
                .await
                .set_toast(format!("paste · {}", optimistic::brief_err(&e)));
        }
    }
}

pub(crate) async fn attach_clipboard_bytes(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
    sid: &str,
) -> bool {
    let Some(bytes) = crate::platform::read_clipboard_png() else {
        return false;
    };
    let b64 = crate::platform::encode_base64(&bytes);
    match client
        .image_attach_bytes(sid, &b64, Some("clipboard.png"))
        .await
    {
        Ok(v) if v.get("attached").and_then(|x| x.as_bool()) == Some(true) => {
            let path = v.get("path").and_then(|s| s.as_str()).unwrap_or("");
            insert_image_token(textarea, path);
            if !path.is_empty() {
                state
                    .lock()
                    .await
                    .remember_image(std::path::PathBuf::from(path));
            }
            state
                .lock()
                .await
                .set_toast("clipboard image attached · bytes");
            true
        }
        _ => false,
    }
}

pub(crate) fn insert_image_token(textarea: &mut TextArea<'static>, _path: &str) {
    let input = textarea.lines().join("\n");
    let tok = crate::complete::image_token(crate::complete::next_image_index(&input));
    textarea.insert_str(format!("{}{tok} ", paste_lead(textarea)));
}
