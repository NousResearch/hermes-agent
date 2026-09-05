//! Slash dispatch: match on `/name`, then call handlers or the gateway.
use anyhow::Result;
use ratatui_textarea::TextArea;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::optimistic;
use crate::rpc::GatewayClient;
use crate::state::{resolve_spawn_entry, ActiveView, AppState, ArmedKind};

use super::{
    commit_theme, open_theme_picker, reset_prompt, send_user_turn, toggle_work_sidebar, LoopControl,
};

mod gateway;
mod handlers;
mod mcp;
mod media;
mod mouse;
mod refresh;
mod show;
#[allow(unused_imports)] // named crate surface; callers live in keys/mod, not here
pub(crate) use gateway::{
    apply_branch_action, apply_command_dispatch, apply_picker_action, apply_slash_payload,
    branch_confirm, gateway_slash, gateway_slash_depth, handle_branch_click,
    handle_picker_key_input, open_branch_picker, open_model_picker, picker_confirm, refresh_models,
    set_model, BranchAction, DispatchCtx, PickerAction,
};
#[allow(unused_imports)]
pub(crate) use handlers::{
    apply_vim, configure_tools, create_profile, cycle_permission_mode, delete_stored_session,
    disconnect_model, draft_commit_message, fork_session, inspect_skill, install_skill,
    maybe_auto_resume, move_workspace, peek_profile, react_last, redirect_turn, refresh_plugins,
    refresh_tools, request_handoff, run_bang, run_cli, save_replay, save_session, search_skills,
    selected_skill_name, selected_toolset_name, set_personality, set_session_hidden,
    steer_or_queue, toggle_plugin, undo_last_exchange,
};
#[allow(unused_imports)]
pub(crate) use mcp::{
    apply_memory_edit, begin_memory_edit, cron_action, delete_memory, handle_mcp_key_input,
    mcp_add, mcp_oauth_login, mcp_remove, mcp_test, peek_memory, peek_toolset, refresh_cron,
    refresh_mcp, reload_mcp, selected_cron_id, selected_mcp, selected_memory, selected_plugin,
    selected_profile_name, selected_provider_slug,
};
#[allow(unused_imports)]
pub(crate) use media::{
    attach_clipboard_bytes, attach_named_file, attach_named_image, attach_named_pdf, detach_image,
    imagine_image, insert_image_token, paste_clipboard_image, replay_diff,
};
#[allow(unused_imports)]
pub(crate) use mouse::{click_composer, click_queue, handle_mouse, QueueClick};
#[allow(unused_imports)]
pub(crate) use refresh::{
    agent_peek_body, apply_bg_action, apply_detect_drop, apply_rollback, background_confirm,
    detect_drop_paste, interrupt_selected_subagent, interrupt_subagent_ids, kill_selected_process,
    load_processes, load_rollback_diff, persist_paste_collapse, refresh_agents, refresh_catalog,
    refresh_context, refresh_memory, refresh_profiles, refresh_rollback, refresh_sessions,
    refresh_skills, refresh_slash_complete, selected_agent, selected_agent_id, set_agents_paused,
    start_background, steer_selected_subagent, stop_background_processes, stop_dock_entry,
    BgAction,
};
#[allow(unused_imports)]
pub(crate) use show::{
    browser_command, change_cwd, config_key, config_value_text, drill_project, load_replay,
    open_projects_overlay, scan_projects, set_or_show_title, show_config, show_credits, show_facts,
    show_insights, show_logs, show_projects, show_replay, show_setup, show_stored_history,
    show_verify,
};

pub(crate) async fn dispatch_slash(
    name: &str,
    arg: &str,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> Result<LoopControl> {
    match name {
        "/exit" | "/quit" => {
            let mut s = state.lock().await;
            if s.has_unsaved("") && !s.take_armed(ArmedKind::Quit) {
                s.arm(ArmedKind::Quit, "queued prompts · /exit again to quit");
                return Ok(LoopControl::Continue);
            }
            return Ok(LoopControl::Quit);
        }
        "/help" => {
            let mut s = state.lock().await;
            s.active_view = ActiveView::Help;
            s.slash_open = false;
            s.mark_dirty();
        }
        "/copy" => {
            let mut s = state.lock().await;
            if arg.is_empty() {
                if !s.copy_latest_response() {
                    s.set_toast("nothing to copy");
                }
            } else {
                match arg.trim().parse::<usize>() {
                    Ok(n) => {
                        if !s.copy_assistant(Some(n)) {
                            s.set_toast("nothing to copy");
                        }
                    }
                    Err(_) => s.set_toast("usage: /copy [number]"),
                }
            }
        }
        "/history" => {
            let stored = matches!(
                arg.trim().to_ascii_lowercase().as_str(),
                "stored" | "remote" | "db" | "gateway"
            );
            if stored {
                show_stored_history(state, client).await;
            } else {
                let preview = arg.trim().parse::<usize>().unwrap_or(400);
                let empty = {
                    let mut s = state.lock().await;
                    match crate::shell::history_text(&s.messages, preview) {
                        Some(body) => {
                            s.open_peek("history".into(), body, None);
                            false
                        }
                        None => true,
                    }
                };
                if empty {
                    show_stored_history(state, client).await;
                }
            }
        }
        "/status" => {
            let sid = state.lock().await.session_id.clone();
            let Some(sid) = sid else {
                state.lock().await.set_toast("no session");
                return Ok(LoopControl::Continue);
            };
            match client.session_status(&sid).await {
                Ok(v) => {
                    let body = v
                        .get("output")
                        .and_then(|x| x.as_str())
                        .unwrap_or("(no status)")
                        .to_string();
                    state.lock().await.open_peek("status".into(), body, None);
                }
                Err(e) => {
                    state
                        .lock()
                        .await
                        .set_toast(format!("status · {}", optimistic::brief_err(&e)));
                }
            }
        }
        "/title" => {
            set_or_show_title(state, client, arg).await;
        }
        "/pwd" => {
            let cwd = state.lock().await.metrics.cwd.clone();
            state.lock().await.set_toast(cwd);
        }
        "/cd" => {
            change_cwd(state, client, arg).await;
        }
        "/steer" => {
            steer_or_queue(state, client, arg).await;
        }
        "/redirect" => {
            redirect_turn(state, client, arg).await;
        }
        "/workspace" => {
            move_workspace(state, client, arg).await;
        }
        "/projects" => {
            show_projects(state, client, arg).await;
        }
        "/cli" => {
            run_cli(state, client, arg).await;
        }
        "/queue" => {
            let mut s = state.lock().await;
            if arg.is_empty() {
                let n = s.prompt_queue.len();
                s.set_toast(format!("{n} queued"));
            } else {
                s.enqueue(arg.to_string());
                s.set_toast("queued");
            }
        }
        "/recap" => {
            let mut s = state.lock().await;
            match crate::shell::recap(&s.messages) {
                Some(text) => s.add_system(text),
                None => s.set_toast("nothing to recap"),
            }
        }
        "/fast" => {
            config_key(state, client, "fast", arg).await;
        }
        "/reasoning" => {
            config_key(state, client, "reasoning", arg).await;
        }
        "/busy" => {
            config_key(state, client, "busy", arg).await;
        }
        "/verbose" => {
            let v = if arg.is_empty() { "cycle" } else { arg };
            config_key(state, client, "verbose", v).await;
        }
        "/density" => {
            let a = if arg.trim().is_empty() { "toggle" } else { arg };
            config_key(state, client, "density", a).await;
        }
        "/personality" => {
            set_personality(state, client, arg).await;
        }
        "/reload" => match client.reload_env().await {
            Ok(v) => {
                let n = v.get("updated").and_then(|x| x.as_u64()).unwrap_or(0);
                state
                    .lock()
                    .await
                    .set_toast(format!("reloaded .env ({n} vars)"));
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("reload · {}", optimistic::brief_err(&e)));
            }
        },
        "/reload-skills" => match client.skills_reload().await {
            Ok(v) => {
                let body = v
                    .get("output")
                    .and_then(|x| x.as_str())
                    .unwrap_or("skills reloaded")
                    .to_string();
                state
                    .lock()
                    .await
                    .open_peek("skills reload".into(), body, None);
                refresh_catalog(state, client).await;
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("reload-skills · {}", optimistic::brief_err(&e)));
            }
        },
        "/battery" => match client.system_battery().await {
            Ok(v) => {
                let msg = if v.get("available").and_then(|x| x.as_bool()) != Some(true) {
                    "no battery on this machine".into()
                } else {
                    let pct = v.get("percent").and_then(|x| x.as_u64()).unwrap_or(0);
                    let plug = if v.get("plugged").and_then(|x| x.as_bool()) == Some(true) {
                        "charging"
                    } else {
                        "battery"
                    };
                    format!("{plug} {pct}%")
                };
                state.lock().await.set_toast(msg);
            }
            Err(e) => {
                state
                    .lock()
                    .await
                    .set_toast(format!("battery · {}", optimistic::brief_err(&e)));
            }
        },
        "/image" => {
            let a = arg.trim();
            let lower = a.to_ascii_lowercase();
            if matches!(lower.as_str(), "detach" | "clear" | "remove")
                || lower.starts_with("detach ")
            {
                let path = a
                    .split_once(char::is_whitespace)
                    .map(|x| x.1)
                    .unwrap_or("")
                    .trim();
                detach_image(state, client, textarea, path).await;
            } else {
                attach_named_image(state, client, textarea, arg).await;
            }
        }
        "/paste" => {
            paste_clipboard_image(state, client, textarea).await;
        }
        "/credits" => {
            show_credits(state, client).await;
        }
        "/mem" => {
            let body = crate::platform::process_mem();
            state.lock().await.open_peek("mem".into(), body, None);
        }
        "/mouse" => {
            return Ok(LoopControl::MouseToggle);
        }
        "/cron" => {
            let mut s = state.lock().await;
            s.active_view = ActiveView::Cron;
            s.modal_selected = 0;
            s.picker_filter.clear();
            s.mark_dirty();
            drop(s);
            refresh_cron(state, client).await;
        }
        "/setup" => {
            show_setup(state, client).await;
        }
        "/config" => {
            show_config(state, client).await;
        }
        "/facts" => {
            show_facts(state, client).await;
        }
        "/verify" | "/review" => {
            show_verify(state, client).await;
        }
        "/replay" => {
            let a = arg.trim();
            let lower = a.to_ascii_lowercase();
            if let Some(path) = a.strip_prefix("load ") {
                load_replay(state, client, path.trim()).await;
            } else if a == "save" || a.starts_with("save ") {
                let label = a.strip_prefix("save").unwrap_or("").trim();
                save_replay(state, client, label).await;
            } else if lower == "list" || lower == "ls" || a.is_empty() {
                show_replay(state, client).await;
            } else {
                show_replay(state, client).await;
                let path = {
                    let s = state.lock().await;
                    if lower == "last" {
                        s.spawn_trees.first().map(|e| e.path.clone())
                    } else {
                        resolve_spawn_entry(a, &s.spawn_trees).map(|e| e.path.clone())
                    }
                };
                if let Some(path) = path {
                    load_replay(state, client, &path).await;
                } else {
                    state
                        .lock()
                        .await
                        .set_toast(format!("replay: no entry {a} · /replay lists indexes"));
                }
            }
        }
        "/replay-diff" => {
            replay_diff(state, client, arg).await;
        }
        "/hide" => {
            let id = arg.trim();
            if id.is_empty() {
                state.lock().await.set_toast("usage: /hide <session_id>");
            } else {
                set_session_hidden(state, client, id, true).await;
            }
        }
        "/unhide" => {
            let id = arg.trim();
            if id.is_empty() {
                state.lock().await.set_toast("usage: /unhide <session_id>");
            } else {
                set_session_hidden(state, client, id, false).await;
            }
        }
        "/react" => {
            react_last(state, client, arg).await;
        }
        "/imagine" => {
            imagine_image(state, client, arg).await;
        }
        "/insights" => {
            show_insights(state, client).await;
        }
        "/indicator" => {
            config_key(state, client, "indicator", arg).await;
        }
        "/statusbar" => {
            let a = if arg.trim().is_empty() { "toggle" } else { arg };
            config_key(state, client, "statusbar", a).await;
        }
        "/redraw" => {
            state.lock().await.mark_dirty();
            state.lock().await.set_toast("redraw");
        }
        "/file" => {
            attach_named_file(state, client, textarea, arg).await;
        }
        "/pdf" => {
            attach_named_pdf(state, client, textarea, arg).await;
        }
        "/delete" => {
            let id = if arg.trim().is_empty() {
                None
            } else {
                Some(arg.trim().to_string())
            };
            if let Some(id) = id {
                delete_stored_session(state, client, &id).await;
            } else {
                state.lock().await.set_toast("usage: /delete <session_id>");
            }
        }
        "/logs" => {
            show_logs(state).await;
        }
        "/fortune" => {
            let daily = matches!(
                arg.trim().to_ascii_lowercase().as_str(),
                "daily" | "today" | "stable"
            );
            let tip = if daily {
                let day = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() / 86400)
                    .unwrap_or(0) as usize;
                crate::tips::TIPS[day % crate::tips::COUNT]
            } else {
                crate::tips::TIPS[crate::tips::start_index()]
            };
            state.lock().await.add_system(tip);
        }
        "/theme-info" => {
            let mut s = state.lock().await;
            let id = s.theme_id.clone();
            let blurb = crate::ui::theme::catalog()
                .iter()
                .find(|p| p.id == id)
                .map(|p| p.blurb)
                .unwrap_or("");
            s.set_toast(if blurb.is_empty() {
                format!("theme {id}")
            } else {
                format!("theme {id} · {blurb}")
            });
        }
        "/update" => {
            state
                .lock()
                .await
                .set_toast("quit the TUI, then: hermes update");
        }
        "/open" => {
            state.lock().await.open_latest_media();
        }
        "/context" | "/usage" => {
            let mut s = state.lock().await;
            s.active_view = ActiveView::Context;
            s.mark_dirty();
            drop(s);
            refresh_context(state, client).await;
        }
        "/theme" | "/skin" => {
            let mut s = state.lock().await;
            if arg.is_empty() {
                open_theme_picker(&mut s);
            } else {
                commit_theme(&mut s, arg);
            }
        }
        "/diff" => {
            let mut s = state.lock().await;
            if s.diff_tool_id.is_some() {
                s.close_edit_diff();
            } else {
                s.split_diff = !s.split_diff;
                if s.split_diff {
                    s.split_work = false;
                    s.work_focus = false;
                    s.refresh_diff();
                }
            }
            s.mark_dirty();
        }
        "/files" | "/explorer" => {
            let mut s = state.lock().await;
            s.split_files = !s.split_files;
            if s.split_files {
                s.files_focus = true;
                s.trace_focus = false;
                s.split_work = false;
                s.work_focus = false;
                s.refresh_files();
            } else {
                s.files_focus = false;
                s.files_list = None;
            }
            s.mark_dirty();
        }
        "/work" => {
            toggle_work_sidebar(state, client).await;
        }
        "/overview" | "/tasks" => {
            let mut s = state.lock().await;
            s.active_view = ActiveView::Tasks;
            s.modal_selected = 0;
            s.mark_dirty();
        }
        "/model" => {
            if arg.is_empty() {
                open_model_picker(state, client).await;
            } else {
                let mut parts = arg.splitn(2, char::is_whitespace);
                let first = parts.next().unwrap_or("");
                if first.eq_ignore_ascii_case("disconnect") || first.eq_ignore_ascii_case("forget")
                {
                    let slug = parts.next().unwrap_or("").trim();
                    if slug.is_empty() {
                        state
                            .lock()
                            .await
                            .set_toast("usage: /model disconnect <slug>");
                    } else {
                        disconnect_model(state, client, slug).await;
                    }
                } else {
                    set_model(state, client, arg, "").await;
                }
            }
        }
        "/branches" => {
            open_branch_picker(state).await;
        }
        "/skills" => {
            let a = arg.trim();
            if a.is_empty() {
                let mut s = state.lock().await;
                s.active_view = ActiveView::Skills;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
                drop(s);
                refresh_skills(state, client).await;
            } else {
                let mut parts = a.splitn(2, char::is_whitespace);
                let sub = parts.next().unwrap_or("").to_ascii_lowercase();
                let rest = parts.next().unwrap_or("").trim();
                match sub.as_str() {
                    "inspect" | "show" => {
                        if rest.is_empty() {
                            state
                                .lock()
                                .await
                                .set_toast("usage: /skills inspect <name>");
                        } else {
                            inspect_skill(state, client, rest).await;
                        }
                    }
                    "install" | "add" => {
                        if rest.is_empty() {
                            state
                                .lock()
                                .await
                                .set_toast("usage: /skills install <name>");
                        } else {
                            install_skill(state, client, rest).await;
                        }
                    }
                    "search" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /skills search <q>");
                        } else {
                            search_skills(state, client, rest).await;
                        }
                    }
                    _ => {
                        inspect_skill(state, client, a).await;
                    }
                }
            }
        }
        "/commit" => {
            draft_commit_message(state, client).await;
        }
        "/vim" => {
            let mut s = state.lock().await;
            if s.vim.is_some() {
                s.vim = None;
                s.set_toast("vim off · esc is interrupt");
            } else {
                s.vim = Some(crate::composer_vim::VimState::normal());
                s.set_toast("vim normal · i insert  dd dw gg  esc leaves");
            }
        }
        "/motion" => {
            let mut s = state.lock().await;
            let a = arg.trim().to_ascii_lowercase();
            if matches!(a.as_str(), "status" | "show" | "?") {
                s.set_toast(if crate::ui::motion::enabled() {
                    "motion on"
                } else {
                    "motion off"
                });
            } else {
                let next = match a.as_str() {
                    "on" | "true" | "1" | "yes" => {
                        crate::ui::motion::set_enabled(true);
                        true
                    }
                    "off" | "false" | "0" | "no" => {
                        crate::ui::motion::set_enabled(false);
                        false
                    }
                    _ => crate::ui::motion::toggle(),
                };
                s.set_toast(if next { "motion on" } else { "motion off" });
            }
        }
        "/handoff" => {
            request_handoff(state, client, arg).await;
        }
        "/profiles" | "/bots" => {
            let a = arg.trim();
            if a.is_empty() {
                let mut s = state.lock().await;
                s.active_view = ActiveView::Profiles;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
                drop(s);
                refresh_profiles(state, client).await;
            } else {
                let mut parts = a.splitn(2, char::is_whitespace);
                let sub = parts.next().unwrap_or("").to_ascii_lowercase();
                let rest = parts.next().unwrap_or("").trim();
                match sub.as_str() {
                    "new" | "create" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /profiles new <slug>");
                        } else {
                            create_profile(state, client, rest, None).await;
                        }
                    }
                    "clone" => {
                        let mut bits = rest.splitn(2, char::is_whitespace);
                        let name = bits.next().unwrap_or("").trim();
                        let src = bits.next().unwrap_or("").trim();
                        if name.is_empty() {
                            state
                                .lock()
                                .await
                                .set_toast("usage: /profiles clone <slug> [from]");
                        } else {
                            let from = if src.is_empty() { None } else { Some(src) };
                            create_profile(state, client, name, from).await;
                        }
                    }
                    _ => {
                        peek_profile(state, client, a).await;
                    }
                }
            }
        }
        "/agents" => {
            let sub = arg.trim().to_ascii_lowercase();
            match sub.as_str() {
                "pause" => set_agents_paused(state, client, true).await,
                "resume" | "unpause" => set_agents_paused(state, client, false).await,
                "status" => {
                    refresh_agents(state, client).await;
                    let s = state.lock().await;
                    let paused = if s.agents_paused { "paused" } else { "active" };
                    let caps = s.agents_caps.clone();
                    drop(s);
                    state.lock().await.set_toast(if caps.is_empty() {
                        format!("delegation · {paused}")
                    } else {
                        format!("delegation · {paused} · {caps}")
                    });
                }
                _ => {
                    let mut s = state.lock().await;
                    s.agents_replay = false;
                    s.active_view = ActiveView::Agents;
                    s.modal_selected = 0;
                    s.picker_filter.clear();
                    s.mark_dirty();
                    drop(s);
                    refresh_agents(state, client).await;
                }
            }
        }
        "/processes" => {
            toggle_work_sidebar(state, client).await;
        }
        "/stop" => {
            stop_background_processes(state, client).await;
        }
        "/memory" | "/journey" => {
            let mut s = state.lock().await;
            s.active_view = ActiveView::Memory;
            s.modal_selected = 0;
            s.picker_filter.clear();
            s.mark_dirty();
            drop(s);
            refresh_memory(state, client).await;
        }
        "/background" | "/bg" | "/btw" => {
            if !arg.is_empty() {
                start_background(state, client, arg).await;
            }
            let mut s = state.lock().await;
            s.open_background();
            if !arg.is_empty() && !s.bg_tasks.is_empty() {
                s.modal_selected = 1;
            }
        }
        "/rollback" => {
            let mut s = state.lock().await;
            s.active_view = ActiveView::Rollback;
            s.modal_selected = 0;
            s.picker_filter.clear();
            s.rollback_diff.clear();
            s.mark_dirty();
            drop(s);
            refresh_rollback(state, client).await;
        }
        "/sessions" => {
            let mut s = state.lock().await;
            s.active_view = ActiveView::Sessions;
            s.modal_selected = 0;
            s.picker_filter.clear();
            s.mark_dirty();
            drop(s);
            refresh_sessions(state, client).await;
        }
        "/trace" => {
            let mut s = state.lock().await;
            s.split_trace = !s.split_trace;
            s.trace_focus = false;
            s.mark_dirty();
        }
        "/tips" => {
            let mut s = state.lock().await;
            let open = !s.tips_open;
            s.set_tips_open(open);
            s.set_toast(if open {
                "tips on"
            } else {
                "tips hidden · /tips"
            });
        }
        "/thinking" => {
            let mut s = state.lock().await;
            s.show_thinking = !s.show_thinking;
            s.mark_dirty();
        }
        "/clear" => {
            let mut s = state.lock().await;
            let msg = s.clear_transcript();
            s.set_toast_for(msg, crate::optimistic::undo_ttl());
        }
        "/goal" => {
            let mut s = state.lock().await;
            if arg.is_empty() {
                let current = s.goal.clone().unwrap_or_else(|| "(none)".into());
                s.add_system(format!("Goal: {current}"));
            } else {
                s.goal = Some(arg.to_string());
                s.set_toast("Goal set");
            }
        }
        "/yolo" => {
            let mut s = state.lock().await;
            let (prev, epoch) = optimistic::apply_yolo_toggle(&mut s);
            let next = s.metrics.permission_mode;
            let sid = s.session_id.clone();
            drop(s);
            if let Some(sid) = sid {
                optimistic::spawn_mode_reconcile(state, client, sid, epoch, prev, next);
            }
        }
        "/plan" => {
            let mut s = state.lock().await;
            let (prev, next, epoch) = optimistic::apply_plan_mode(&mut s);
            let sid = s.session_id.clone();
            drop(s);
            if let Some(sid) = sid {
                optimistic::spawn_mode_reconcile(state, client, sid, epoch, prev, next);
            }
        }
        "/mcp" => {
            let a = arg.trim();
            if a.is_empty() {
                let mut s = state.lock().await;
                s.active_view = ActiveView::Mcp;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
                drop(s);
                refresh_mcp(state, client).await;
            } else {
                let mut parts = a.splitn(2, char::is_whitespace);
                let sub = parts.next().unwrap_or("").to_ascii_lowercase();
                let rest = parts.next().unwrap_or("").trim();
                match sub.as_str() {
                    "add" | "install" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /mcp add <name>");
                        } else {
                            mcp_add(state, client, rest).await;
                        }
                    }
                    "remove" | "rm" | "delete" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /mcp remove <name>");
                        } else {
                            mcp_remove(state, client, rest).await;
                        }
                    }
                    "test" | "probe" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /mcp test <name>");
                        } else {
                            mcp_test(state, client, rest).await;
                        }
                    }
                    "reload" => reload_mcp(state, client).await,
                    "login" | "oauth" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /mcp login <name>");
                        } else {
                            mcp_oauth_login(state, client, rest).await;
                        }
                    }
                    "key" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /mcp key <name>");
                        } else {
                            let mut s = state.lock().await;
                            s.active_view = ActiveView::Mcp;
                            s.mcp_key_name = Some(rest.to_string());
                            s.picker_key.clear();
                            s.picker_key_error.clear();
                            s.set_toast(format!("mcp key for {rest} · paste · enter"));
                        }
                    }
                    _ => state.lock().await.set_toast(
                        "usage: /mcp [add|remove|test|key|login <name>]  or  /mcp reload",
                    ),
                }
            }
        }
        "/reload-mcp" => {
            reload_mcp(state, client).await;
        }
        "/palette" => {
            let mut s = state.lock().await;
            s.active_view = ActiveView::Palette;
            s.modal_selected = 0;
            s.picker_filter.clear();
            s.mark_dirty();
        }
        "/init" => {
            let cwd = state.lock().await.metrics.cwd.clone();
            match crate::shell::init_agents_md(&cwd) {
                Ok(path) => state.lock().await.add_system(format!("wrote {path}")),
                Err(e) => {
                    let mut s = state.lock().await;
                    let path = crate::shell::agents_md_path(&cwd);
                    if path.exists() {
                        match std::fs::read_to_string(&path) {
                            Ok(body) => s.open_peek("AGENTS.md".into(), body, None),
                            Err(_) => s.set_toast(e),
                        }
                    } else {
                        s.set_toast(e);
                    }
                }
            }
        }
        "/export" => {
            let mut s = state.lock().await;
            let md = crate::shell::transcript_markdown(&s.messages);
            match crate::platform::copy_to_clipboard(&md) {
                Ok(()) => s.set_toast("copied transcript markdown"),
                Err(e) => s.set_toast(format!("export failed: {e}")),
            }
        }
        "/fork" | "/branch" => {
            fork_session(state, client, arg).await;
        }
        "/undo" => {
            undo_last_exchange(state, client).await;
        }
        "/save" => {
            save_session(state, client).await;
        }
        "/editor" | "/prompt" => {
            return Ok(LoopControl::Editor);
        }
        "/focus" => {
            let mut s = state.lock().await;
            let next = match arg.trim().to_ascii_lowercase().as_str() {
                "on" | "true" | "1" => true,
                "off" | "false" | "0" => false,
                "status" | "show" | "?" => s.focus_view,
                _ => !s.focus_view,
            };
            s.focus_view = next;
            s.set_toast(if next {
                "focus on · last turn only"
            } else {
                "focus off"
            });
        }
        "/tools" => {
            if arg.is_empty() {
                let mut s = state.lock().await;
                s.active_view = ActiveView::Tools;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
                drop(s);
                refresh_tools(state, client).await;
            } else {
                let mut parts = arg.split_whitespace();
                let sub = parts.next().unwrap_or("").to_ascii_lowercase();
                if sub == "enable" || sub == "disable" {
                    let names: Vec<String> = parts.map(|s| s.to_string()).collect();
                    if names.is_empty() {
                        state
                            .lock()
                            .await
                            .set_toast(format!("usage: /tools {sub} <name> [name …]"));
                    } else {
                        configure_tools(state, client, &sub, names).await;
                    }
                } else {
                    let cmd = format!("/tools {arg}");
                    return gateway_slash(state, client, textarea, &cmd).await;
                }
            }
        }
        "/plugins" => {
            let a = arg.trim();
            if a.is_empty() {
                let mut s = state.lock().await;
                s.active_view = ActiveView::Plugins;
                s.modal_selected = 0;
                s.picker_filter.clear();
                s.mark_dirty();
                drop(s);
                refresh_plugins(state, client).await;
            } else {
                let mut parts = a.splitn(2, char::is_whitespace);
                let sub = parts.next().unwrap_or("").to_ascii_lowercase();
                let rest = parts.next().unwrap_or("").trim();
                match sub.as_str() {
                    "enable" | "on" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /plugins enable <key>");
                        } else {
                            toggle_plugin(state, client, rest, true).await;
                        }
                    }
                    "disable" | "off" => {
                        if rest.is_empty() {
                            state
                                .lock()
                                .await
                                .set_toast("usage: /plugins disable <key>");
                        } else {
                            toggle_plugin(state, client, rest, false).await;
                        }
                    }
                    "toggle" => {
                        if rest.is_empty() {
                            state.lock().await.set_toast("usage: /plugins toggle <key>");
                        } else {
                            let enabled = state
                                .lock()
                                .await
                                .plugins
                                .iter()
                                .find(|p| p.key == rest || p.name == rest)
                                .map(|p| p.enabled)
                                .unwrap_or(false);
                            toggle_plugin(state, client, rest, !enabled).await;
                        }
                    }
                    _ => {
                        let cmd = format!("/plugins {a}");
                        return gateway_slash(state, client, textarea, &cmd).await;
                    }
                }
            }
        }
        "/sandbox" => {
            let s = state.lock().await;
            let backend = s.metrics.terminal_backend.trim();
            let mode = s.metrics.approval_mode.trim();
            let backend = if backend.is_empty() { "local" } else { backend };
            let msg = if mode.is_empty() {
                format!("terminal {backend}")
            } else {
                format!("terminal {backend} · approvals {mode}")
            };
            drop(s);
            state.lock().await.set_toast(msg);
        }
        "/retry" => {
            let text = state.lock().await.last_user_text();
            let Some(text) = text else {
                state.lock().await.set_toast("nothing to retry");
                return Ok(LoopControl::Continue);
            };
            return send_user_turn(text, state, client, textarea).await;
        }
        "/details" => {
            state.lock().await.toggle_details();
        }
        "/browser" => {
            browser_command(state, client, arg).await;
        }
        other => {
            return gateway_slash(state, client, textarea, &format!("{other} {arg}")).await;
        }
    }
    *textarea = reset_prompt(false);
    Ok(LoopControl::Continue)
}
