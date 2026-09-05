//! Fall-through slash.exec / command.dispatch, plus model/branch pickers.
use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui_textarea::TextArea;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::slash::{
    is_dispatch_routing_noise, parse_command_dispatch, parse_slash, CommandDispatch, SlashKind,
};
use crate::state::{
    parse_model_providers, parse_saved_provider, ActiveView, AppState, PickerStage,
};

use super::super::{
    is_compose_char, reset_prompt, set_prompt_text, submit_dispatch_message, LoopControl,
};
use super::dispatch_slash;

pub(crate) async fn gateway_slash(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
    command: &str,
) -> Result<LoopControl> {
    gateway_slash_depth(state, client, textarea, command, 0).await
}

pub(crate) async fn gateway_slash_depth(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
    command: &str,
    depth: u8,
) -> Result<LoopControl> {
    let cmd = command.trim().to_string();
    let sid = state.lock().await.session_id.clone();
    let Some(sid) = sid else {
        state.lock().await.add_system("no active session");
        *textarea = reset_prompt(false);
        return Ok(LoopControl::Continue);
    };
    let (name, arg) = parse_slash(&cmd);
    let ctx = DispatchCtx {
        name: &name,
        arg: &arg,
        depth,
    };
    match client.slash_exec(&sid, &cmd).await {
        Ok(v) => apply_slash_payload(state, client, textarea, ctx, v, &cmd).await,
        Err(e) => {
            let exec_err = e.to_string();
            match client.command_dispatch(&sid, &name, &arg).await {
                Ok(v) => apply_slash_payload(state, client, textarea, ctx, v, &cmd).await,
                Err(de) => {
                    if cmd.contains("compress") {
                        state.lock().await.set_toast("compressing context…");
                        optimistic::spawn_compress(state, client, sid);
                        *textarea = reset_prompt(false);
                        return Ok(LoopControl::Continue);
                    }
                    let dispatch_err = de.to_string();
                    let msg = if is_dispatch_routing_noise(&dispatch_err) {
                        exec_err
                    } else {
                        dispatch_err
                    };
                    state
                        .lock()
                        .await
                        .add_system(format!("/{name} failed: {msg}"));
                    *textarea = reset_prompt(false);
                    Ok(LoopControl::Continue)
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DispatchCtx<'a> {
    name: &'a str,
    arg: &'a str,
    depth: u8,
}

pub(crate) async fn apply_slash_payload(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
    ctx: DispatchCtx<'_>,
    v: serde_json::Value,
    cmd: &str,
) -> Result<LoopControl> {
    if let Some(dispatch) = parse_command_dispatch(&v) {
        return apply_command_dispatch(state, client, textarea, ctx, dispatch).await;
    }
    if let Some(out) = v.get("output").and_then(|o| o.as_str()) {
        if !out.trim().is_empty() {
            state.lock().await.add_system(out);
            *textarea = reset_prompt(false);
            return Ok(LoopControl::Continue);
        }
    }
    if cmd.contains("compress") {
        let sid = state.lock().await.session_id.clone();
        state.lock().await.set_toast("compressing context…");
        if let Some(sid) = sid {
            optimistic::spawn_compress(state, client, sid);
        }
    } else {
        state.lock().await.add_system(v.to_string());
    }
    *textarea = reset_prompt(false);
    Ok(LoopControl::Continue)
}

pub(crate) async fn apply_command_dispatch(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
    ctx: DispatchCtx<'_>,
    dispatch: CommandDispatch,
) -> Result<LoopControl> {
    match dispatch {
        CommandDispatch::Exec { output } | CommandDispatch::Plugin { output } => {
            let body = if output.trim().is_empty() {
                "(no output)"
            } else {
                output.as_str()
            };
            state.lock().await.add_system(body);
            *textarea = reset_prompt(false);
            Ok(LoopControl::Continue)
        }
        CommandDispatch::Alias { target } => {
            if ctx.depth >= 4 {
                state.lock().await.set_toast("alias loop");
                *textarea = reset_prompt(false);
                return Ok(LoopControl::Continue);
            }
            let target = target.trim().trim_start_matches('/');
            let line = if ctx.arg.is_empty() {
                format!("/{target}")
            } else {
                format!("/{target} {}", ctx.arg)
            };
            let (n, a) = parse_slash(&line);
            let full = format!("/{n}");
            let catalog = state.lock().await.slash_catalog.clone();
            if catalog
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(&full) && c.kind == SlashKind::Local)
            {
                return Box::pin(dispatch_slash(&full, &a, state, client, textarea)).await;
            }
            Box::pin(gateway_slash_depth(
                state,
                client,
                textarea,
                &line,
                ctx.depth + 1,
            ))
            .await
        }
        CommandDispatch::Prefill { message, notice } => {
            if !notice.trim().is_empty() {
                state.lock().await.add_system(notice.trim());
            }
            if !message.is_empty() {
                set_prompt_text(textarea, &message);
            } else {
                *textarea = reset_prompt(false);
            }
            Ok(LoopControl::Continue)
        }
        CommandDispatch::Send {
            message,
            notice,
            display,
        } => {
            if !notice.trim().is_empty() {
                state.lock().await.add_system(notice.trim());
            }
            if message.trim().is_empty() {
                state
                    .lock()
                    .await
                    .add_system(format!("/{}: empty message", ctx.name));
                *textarea = reset_prompt(false);
                return Ok(LoopControl::Continue);
            }
            let visible = if display.trim().is_empty() {
                format!("/{}", ctx.name)
            } else {
                display
            };
            submit_dispatch_message(visible, message, state, client, textarea).await
        }
        CommandDispatch::Skill {
            name: skill,
            message,
            display,
        } => {
            if message.trim().is_empty() {
                state
                    .lock()
                    .await
                    .add_system(format!("/{}: skill payload missing message", ctx.name));
                *textarea = reset_prompt(false);
                return Ok(LoopControl::Continue);
            }
            let visible = if display.trim().is_empty() {
                format!("/{skill}")
            } else {
                display
            };
            submit_dispatch_message(visible, message, state, client, textarea).await
        }
    }
}

pub(crate) async fn set_model(
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    model: &str,
    provider: &str,
) {
    let mut s = state.lock().await;
    let Some(sid) = s.session_id.clone() else {
        return;
    };
    let prev_model = s.metrics.active_model.clone();
    let prev_provider = s.metrics.active_provider.clone();
    s.metrics.active_model = model.to_string();
    if !provider.is_empty() {
        s.metrics.active_provider = provider.to_string();
    }
    s.model_epoch = s.model_epoch.wrapping_add(1);
    let epoch = s.model_epoch;
    let toast = if provider.is_empty() {
        format!("model  {model}")
    } else {
        format!("{provider} · {model}")
    };
    s.set_toast(toast);
    drop(s);
    let value = if provider.is_empty() {
        format!("{model} --session")
    } else {
        format!("{model} --provider {provider} --session")
    };
    optimistic::spawn_model(state, client, sid, epoch, prev_model, prev_provider, value);
}

pub(crate) async fn refresh_models(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    let sid = state.lock().await.session_id.clone().unwrap_or_default();
    match client.model_options(&sid).await {
        Ok(v) => {
            let providers = parse_model_providers(&v);
            let mut s = state.lock().await;
            if let Some(model) = v.get("model").and_then(|m| m.as_str()) {
                if !model.is_empty() {
                    s.metrics.active_model = model.to_string();
                }
            }
            if let Some(provider) = v.get("provider").and_then(|m| m.as_str()) {
                if !provider.is_empty() {
                    s.metrics.active_provider = provider.to_string();
                }
            }
            s.providers = providers;
            if s.picker_stage == PickerStage::Providers {
                if let Some(i) = s.providers.iter().position(|p| p.is_current) {
                    s.modal_selected = i;
                }
            }
            let n = s.picker_len();
            s.clamp_modal(n);
            s.mark_dirty();
        }
        Err(e) => {
            tracing::warn!("model.options failed: {e}");
        }
    }
}

pub(crate) async fn open_model_picker(state: &Mutex<AppState>, client: &Arc<GatewayClient>) {
    {
        let mut s = state.lock().await;
        s.open_picker();
    }
    refresh_models(state, client).await;
}

#[derive(Debug)]
pub(crate) enum PickerAction {
    None,
    SetModel { model: String, provider: String },
    Toast(String),
}

pub(crate) fn picker_confirm(s: &mut AppState) -> PickerAction {
    match s.picker_stage {
        PickerStage::Providers => {
            let Some(&pi) = s.filtered_provider_indices().get(s.modal_selected) else {
                return PickerAction::None;
            };
            let Some(p) = s.providers.get(pi) else {
                return PickerAction::None;
            };
            let authenticated = p.authenticated;
            let warning = p.warning.clone();
            let name = p.name.clone();
            let models = p.models.clone();
            let accepts_key = p.accepts_inline_key();
            if !authenticated {
                if accepts_key {
                    s.open_provider_key(pi);
                    return PickerAction::None;
                }
                let msg = if warning.is_empty() {
                    format!("{name} needs setup — run hermes model")
                } else {
                    warning
                };
                return PickerAction::Toast(msg);
            }
            if models.is_empty() {
                return PickerAction::Toast(format!("{name} has no models"));
            }
            s.picker_provider = pi;
            s.picker_stage = PickerStage::Models;
            s.picker_filter.clear();
            s.modal_selected = models
                .iter()
                .position(|m| *m == s.metrics.active_model || s.metrics.active_model.ends_with(m))
                .unwrap_or(0);
            s.mark_dirty();
            PickerAction::None
        }
        PickerStage::Models => {
            let Some(&mi) = s.filtered_model_indices().get(s.modal_selected) else {
                return PickerAction::None;
            };
            let Some(p) = s.providers.get(s.picker_provider) else {
                return PickerAction::None;
            };
            let Some(model) = p.models.get(mi).cloned() else {
                return PickerAction::None;
            };
            let provider = p.slug.clone();
            s.active_view = ActiveView::Chat;
            s.picker_stage = PickerStage::Providers;
            s.picker_filter.clear();
            s.clear_picker_key();
            s.modal_selected = 0;
            s.mark_dirty();
            PickerAction::SetModel { model, provider }
        }
        PickerStage::Key => PickerAction::None,
    }
}

pub(crate) async fn handle_picker_key_input(
    key: KeyEvent,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
) -> Result<LoopControl> {
    let mut s = state.lock().await;
    if s.picker_key_saving {
        return Ok(LoopControl::Continue);
    }
    match key.code {
        KeyCode::Esc => {
            s.picker_stage = PickerStage::Providers;
            s.clear_picker_key();
            s.modal_selected = s
                .filtered_provider_indices()
                .iter()
                .position(|i| *i == s.picker_provider)
                .unwrap_or(0);
            s.mark_dirty();
        }
        KeyCode::Enter => {
            let api_key = s.picker_key.trim().to_string();
            if api_key.is_empty() {
                s.picker_key_error = "paste an API key".into();
                s.mark_dirty();
                return Ok(LoopControl::Continue);
            }
            let slug = s
                .selected_provider()
                .map(|p| p.slug.clone())
                .unwrap_or_default();
            if slug.is_empty() {
                s.picker_key_error = "no provider selected".into();
                s.mark_dirty();
                return Ok(LoopControl::Continue);
            }
            let sid = s.session_id.clone().unwrap_or_default();
            s.picker_key_saving = true;
            s.picker_key_error.clear();
            s.mark_dirty();
            drop(s);
            match client.model_save_key(&sid, &slug, &api_key).await {
                Ok(v) => {
                    if let Some(provider) = parse_saved_provider(&v) {
                        let name = provider.name.clone();
                        let mut s = state.lock().await;
                        s.apply_saved_provider(provider);
                        s.set_toast(format!("{name} key saved · pick a model"));
                    } else {
                        let mut s = state.lock().await;
                        s.picker_key_saving = false;
                        s.picker_key_error = "saved, but no provider came back".into();
                        s.mark_dirty();
                    }
                }
                Err(e) => {
                    let mut s = state.lock().await;
                    s.picker_key_saving = false;
                    s.picker_key_error = optimistic::brief_err(&e);
                    s.mark_dirty();
                }
            }
        }
        KeyCode::Backspace => {
            s.picker_key.pop();
            s.mark_dirty();
        }
        KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            s.picker_key.clear();
            s.mark_dirty();
        }
        KeyCode::Char(c)
            if !key.modifiers.contains(KeyModifiers::CONTROL) && is_compose_char(c) =>
        {
            s.picker_key.push(c);
            s.picker_key_error.clear();
            s.mark_dirty();
        }
        _ => {}
    }
    Ok(LoopControl::Continue)
}

pub(crate) async fn apply_picker_action(
    action: PickerAction,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
) {
    match action {
        PickerAction::None => {}
        PickerAction::SetModel { model, provider } => {
            set_model(state, client, &model, &provider).await;
        }
        PickerAction::Toast(msg) => {
            state.lock().await.set_toast(msg);
        }
    }
}

pub(crate) enum BranchAction {
    None,
    Close,
    Switch(String),
    MoveCwd { path: String, name: String },
    Toast(String),
}

pub(crate) fn branch_confirm(s: &mut AppState) -> BranchAction {
    let Some(&bi) = s.filtered_branch_indices().get(s.modal_selected) else {
        return BranchAction::None;
    };
    let Some(b) = s.branches.get(bi) else {
        return BranchAction::None;
    };
    if b.current {
        s.active_view = ActiveView::Chat;
        s.mark_dirty();
        return BranchAction::Close;
    }
    if s.is_generating {
        return BranchAction::Toast("session busy — wait for the turn".into());
    }
    let name = b.name.clone();
    let cwd = s.metrics.cwd.clone();
    let other = b
        .worktree
        .as_deref()
        .filter(|p| !crate::platform::same_path(p, &cwd))
        .map(|p| p.to_string());
    s.active_view = ActiveView::Chat;
    s.mark_dirty();
    if let Some(path) = other {
        BranchAction::MoveCwd { path, name }
    } else {
        BranchAction::Switch(name)
    }
}

pub(crate) async fn apply_branch_action(
    action: BranchAction,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) {
    match action {
        BranchAction::None | BranchAction::Close => {}
        BranchAction::Toast(msg) => {
            state.lock().await.set_toast(msg);
        }
        BranchAction::Switch(name) => {
            let cwd = state.lock().await.metrics.cwd.clone();
            match crate::platform::switch_git_branch(&cwd, &name) {
                Ok(()) => {
                    let (_, branch) = crate::platform::probe_git_repo_branch(&cwd);
                    let mut s = state.lock().await;
                    s.metrics.git_branch = branch.or(Some(name.clone()));
                    s.set_toast(format!("branch  {name}"));
                }
                Err(e) => {
                    let brief: String = e
                        .lines()
                        .next()
                        .unwrap_or("git switch failed")
                        .chars()
                        .take(80)
                        .collect();
                    state.lock().await.set_toast(brief);
                }
            }
        }
        BranchAction::MoveCwd { path, name } => {
            let sid = state.lock().await.session_id.clone();
            let Some(sid) = sid else { return };
            match client.set_cwd(&sid, &path).await {
                Ok(_) => {
                    let mut s = state.lock().await;
                    s.metrics.cwd = path;
                    s.metrics.git_branch = Some(name.clone());
                    let (repo, branch) = crate::platform::probe_git_repo_branch(&s.metrics.cwd);
                    s.metrics.git_repo = repo;
                    if branch.is_some() {
                        s.metrics.git_branch = branch;
                    }
                    s.set_toast(format!("worktree  {name}"));
                }
                Err(e) => {
                    state.lock().await.set_toast(format!("cwd set failed: {e}"));
                }
            }
        }
    }
}

pub(crate) async fn open_branch_picker(state: &Mutex<AppState>) {
    let cwd = state.lock().await.metrics.cwd.clone();
    let branches = crate::platform::list_git_branches(&cwd);
    let mut s = state.lock().await;
    s.branches = branches;
    if s.branches.is_empty() {
        s.set_toast("no git branches here");
        return;
    }
    s.open_branch_picker();
}

pub(crate) async fn handle_branch_click(
    col: u16,
    row: u16,
    state: &Mutex<AppState>,
    client: &Arc<GatewayClient>,
) {
    let action = {
        let mut s = state.lock().await;
        if let Some(inner) = s.picker_list {
            if col >= inner.x
                && col < inner.x.saturating_add(inner.width)
                && row >= inner.y
                && row < inner.y.saturating_add(inner.height)
            {
                let idx = s.picker_offset + (row.saturating_sub(inner.y) as usize);
                if idx < s.picker_len() {
                    s.modal_selected = idx;
                    branch_confirm(&mut s)
                } else {
                    BranchAction::None
                }
            } else {
                s.active_view = ActiveView::Chat;
                s.mark_dirty();
                BranchAction::None
            }
        } else {
            BranchAction::None
        }
    };
    apply_branch_action(action, state, client).await;
}
