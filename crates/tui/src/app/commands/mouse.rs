//! Mouse hit-testing for queue, composer, and overlays.
use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui_textarea::{CursorMove, DataCursor, TextArea};
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::rpc::GatewayClient;
use crate::state::{ActiveView, AppState, DockEntry, HoverKind, MessageRole, PickerStage};

use super::super::{
    close_theme_picker, picker_scroll, reset_prompt, send_now, set_prompt_text, toggle_work_sidebar,
};
use super::{
    apply_bg_action, apply_picker_action, background_confirm, cycle_permission_mode,
    handle_branch_click, open_branch_picker, picker_confirm, refresh_context, refresh_models,
    refresh_sessions, stop_dock_entry, BgAction, PickerAction,
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum QueueClick {
    None,
    Edit(usize),
    Send(usize),
    Drop(usize),
}

pub(crate) fn click_queue(
    s: &mut AppState,
    area: ratatui::layout::Rect,
    col: u16,
    row: u16,
) -> QueueClick {
    if let Some((_, kind)) = s.hit_queue.iter().find(|(h, _)| h.contains(col, row)) {
        return match kind {
            HoverKind::QueueSend(i) => QueueClick::Send(*i),
            HoverKind::QueueEdit(i) => QueueClick::Edit(*i),
            HoverKind::QueueDrop(i) => QueueClick::Drop(*i),
            HoverKind::Queue(i) => QueueClick::Edit(*i),
            _ => QueueClick::None,
        };
    }
    if area.height == 0
        || col < area.x
        || col >= area.x.saturating_add(area.width)
        || row < area.y
        || row >= area.y.saturating_add(area.height)
    {
        return QueueClick::None;
    }
    let vis = row.saturating_sub(area.y) as usize;
    let n = s.prompt_queue.len();
    let (start, end) = crate::ui::queue::queue_window(n, s.queue_edit);
    let idx = start + vis;
    if idx >= end || idx >= n {
        return QueueClick::None;
    }
    QueueClick::Edit(idx)
}

pub(crate) fn click_composer(
    textarea: &mut TextArea<'static>,
    area: ratatui::layout::Rect,
    col: u16,
    row: u16,
) -> bool {
    if area.width < 3 || area.height < 3 {
        return false;
    }
    let inner_x = area.x.saturating_add(1);
    let inner_y = area.y.saturating_add(1);
    let inner_w = area.width.saturating_sub(2);
    let inner_h = area.height.saturating_sub(2);
    if col < inner_x
        || row < inner_y
        || col >= inner_x.saturating_add(inner_w)
        || row >= inner_y.saturating_add(inner_h)
    {
        return false;
    }
    textarea.move_cursor(CursorMove::Jump(
        row.saturating_sub(inner_y),
        col.saturating_sub(inner_x),
    ));
    true
}

pub(crate) async fn handle_mouse(
    mouse: MouseEvent,
    state: &Arc<Mutex<AppState>>,
    client: &Arc<GatewayClient>,
    textarea: &mut TextArea<'static>,
) -> bool {
    if !state.lock().await.mouse_on {
        return false;
    }
    match mouse.kind {
        MouseEventKind::Moved | MouseEventKind::Drag(MouseButton::Left) => {
            let mut s = state.lock().await;
            let kind = s.hover_at(mouse.column, mouse.row);
            if let crate::state::HoverKind::Picker(i) = kind {
                if s.modal_selected != i {
                    s.modal_selected = i;
                }
            }
            s.set_hover(kind)
        }
        MouseEventKind::ScrollUp => {
            let mut s = state.lock().await;
            if s.active_view != ActiveView::Chat {
                picker_scroll(&mut s, -1);
                return true;
            }
            if s.files_focus
                || s.files_list.is_some_and(|a| {
                    mouse.column >= a.x && mouse.column < a.x.saturating_add(a.width)
                })
            {
                s.files_move(-1);
                return true;
            }
            s.scroll_older(3);
            true
        }
        MouseEventKind::ScrollDown => {
            let mut s = state.lock().await;
            if s.active_view != ActiveView::Chat {
                picker_scroll(&mut s, 1);
                return true;
            }
            if s.files_focus
                || s.files_list.is_some_and(|a| {
                    mouse.column >= a.x && mouse.column < a.x.saturating_add(a.width)
                })
            {
                s.files_move(1);
                return true;
            }
            s.scroll_newer(3);
            true
        }
        MouseEventKind::Down(MouseButton::Left) => {
            let action = {
                let mut s = state.lock().await;
                if s.hit_tips_close
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    s.set_tips_open(false);
                    s.set_toast("tips hidden · /tips");
                    return true;
                }
                if s.hit_tips_bar
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    s.next_tip();
                    return true;
                }
                if s.active_view == ActiveView::ThemePicker {
                    if let Some(inner) = s.picker_list {
                        if mouse.column >= inner.x
                            && mouse.column < inner.x.saturating_add(inner.width)
                            && mouse.row >= inner.y
                            && mouse.row < inner.y.saturating_add(inner.height)
                        {
                            let idx =
                                s.picker_offset + (mouse.row.saturating_sub(inner.y) as usize);
                            if idx < s.picker_len() {
                                s.modal_selected = idx;
                                close_theme_picker(&mut s, true);
                                return true;
                            }
                        }
                    }
                    close_theme_picker(&mut s, false);
                    return true;
                }
                if s.active_view == ActiveView::ModelPicker {
                    if let Some(inner) = s.picker_list {
                        if mouse.column >= inner.x
                            && mouse.column < inner.x.saturating_add(inner.width)
                            && mouse.row >= inner.y
                            && mouse.row < inner.y.saturating_add(inner.height)
                        {
                            let idx =
                                s.picker_offset + (mouse.row.saturating_sub(inner.y) as usize);
                            if idx < s.picker_len() {
                                s.modal_selected = idx;
                                picker_confirm(&mut s)
                            } else {
                                PickerAction::None
                            }
                        } else {
                            s.active_view = ActiveView::Chat;
                            s.picker_stage = PickerStage::Providers;
                            s.clear_picker_key();
                            s.mark_dirty();
                            PickerAction::None
                        }
                    } else {
                        PickerAction::None
                    }
                } else if s.active_view == ActiveView::BranchPicker {
                    drop(s);
                    handle_branch_click(mouse.column, mouse.row, state, client).await;
                    return true;
                } else if s.active_view == ActiveView::Background {
                    let action = if let Some(inner) = s.picker_list {
                        if mouse.column >= inner.x
                            && mouse.column < inner.x.saturating_add(inner.width)
                            && mouse.row >= inner.y
                            && mouse.row < inner.y.saturating_add(inner.height)
                        {
                            let idx =
                                s.picker_offset + (mouse.row.saturating_sub(inner.y) as usize);
                            if idx < s.picker_len() {
                                s.modal_selected = idx;
                                background_confirm(&mut s)
                            } else {
                                BgAction::None
                            }
                        } else {
                            s.active_view = ActiveView::Chat;
                            s.picker_list = None;
                            s.picker_filter.clear();
                            s.modal_selected = 0;
                            s.mark_dirty();
                            BgAction::None
                        }
                    } else {
                        BgAction::None
                    };
                    drop(s);
                    apply_bg_action(action, state, client).await;
                    return true;
                } else if s.active_view == ActiveView::Agents {
                    if let Some(inner) = s.picker_list {
                        if mouse.column >= inner.x
                            && mouse.column < inner.x.saturating_add(inner.width)
                            && mouse.row >= inner.y
                            && mouse.row < inner.y.saturating_add(inner.height)
                        {
                            let idx =
                                s.picker_offset + (mouse.row.saturating_sub(inner.y) as usize);
                            if idx < s.picker_len() {
                                s.modal_selected = idx;
                                s.mark_dirty();
                            }
                            return true;
                        }
                    }
                    s.active_view = ActiveView::Chat;
                    s.picker_list = None;
                    s.picker_filter.clear();
                    s.agents_steer = false;
                    s.modal_selected = 0;
                    s.mark_dirty();
                    return true;
                } else if s
                    .hit_mode
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    s.ping_click(HoverKind::Mode);
                    drop(s);
                    cycle_permission_mode(state, client).await;
                    return true;
                } else if s
                    .hit_context
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    s.ping_click(HoverKind::Context);
                    s.active_view = ActiveView::Context;
                    s.mark_dirty();
                    drop(s);
                    refresh_context(state, client).await;
                    return true;
                } else if s
                    .hit_session
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    s.ping_click(HoverKind::Session);
                    s.active_view = ActiveView::Sessions;
                    s.modal_selected = 0;
                    s.picker_filter.clear();
                    s.mark_dirty();
                    drop(s);
                    refresh_sessions(state, client).await;
                    return true;
                } else if s
                    .hit_bg
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    s.ping_click(HoverKind::Background);
                    s.open_background();
                    return true;
                } else if s
                    .hit_agents
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                    || s.hit_process
                        .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    let kind = if s
                        .hit_process
                        .is_some_and(|h| h.contains(mouse.column, mouse.row))
                    {
                        HoverKind::Process
                    } else {
                        HoverKind::Agents
                    };
                    s.ping_click(kind);
                    drop(s);
                    toggle_work_sidebar(state, client).await;
                    return true;
                } else if s
                    .hit_model
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    s.ping_click(HoverKind::Model);
                    s.open_picker();
                    drop(s);
                    refresh_models(state, client).await;
                    return true;
                } else if s
                    .hit_branch
                    .is_some_and(|h| h.contains(mouse.column, mouse.row))
                {
                    s.ping_click(HoverKind::Branch);
                    drop(s);
                    open_branch_picker(state).await;
                    return true;
                } else if s.active_view == ActiveView::Chat {
                    if let Some(entry) = s
                        .hit_dock_stop
                        .iter()
                        .find(|(h, _)| h.contains(mouse.column, mouse.row))
                        .map(|(_, e)| *e)
                    {
                        s.ping_click(HoverKind::DockStop(entry));
                        drop(s);
                        stop_dock_entry(state, client, entry).await;
                        return true;
                    }
                    if let Some(entry) = s
                        .hit_dock
                        .iter()
                        .find(|(h, _)| h.contains(mouse.column, mouse.row))
                        .map(|(_, e)| *e)
                    {
                        s.ping_click(HoverKind::Dock(entry));
                        match entry {
                            DockEntry::Agent(ai) => {
                                s.work_selected = ai;
                                s.work_show_diff = false;
                                let open = !s.split_work;
                                s.mark_dirty();
                                drop(s);
                                if open {
                                    toggle_work_sidebar(state, client).await;
                                    let mut s = state.lock().await;
                                    s.work_selected = ai;
                                    s.work_show_diff = false;
                                    s.mark_dirty();
                                }
                            }
                            DockEntry::Bg(bi) => {
                                s.open_background();
                                if bi < s.bg_tasks.len() {
                                    s.modal_selected = bi + 1;
                                }
                                drop(s);
                            }
                        }
                        return true;
                    }
                    if s.hit_jump
                        .is_some_and(|h| h.contains(mouse.column, mouse.row))
                    {
                        s.jump_to_tail();
                        return true;
                    }
                    if let Some(label) = s
                        .hit_pastes
                        .iter()
                        .find(|(h, _)| h.contains(mouse.column, mouse.row))
                        .map(|(_, l)| l.clone())
                    {
                        s.open_bracket(&label);
                        return true;
                    }
                    if let Some(inner) = s.files_list {
                        if mouse.column >= inner.x
                            && mouse.column < inner.x.saturating_add(inner.width)
                            && mouse.row >= inner.y
                            && mouse.row < inner.y.saturating_add(inner.height)
                        {
                            s.files_focus = true;
                            s.trace_focus = false;
                            let idx = s.files_offset + (mouse.row.saturating_sub(inner.y) as usize);
                            if idx < s.files_rows.len() {
                                s.files_selected = idx;
                                s.load_file_preview();
                            }
                            s.mark_dirty();
                            return true;
                        }
                    }
                    if let Some(inner) = s.work_list {
                        if mouse.column >= inner.x
                            && mouse.column < inner.x.saturating_add(inner.width)
                            && mouse.row >= inner.y
                            && mouse.row < inner.y.saturating_add(inner.height)
                        {
                            s.work_focus = true;
                            s.files_focus = false;
                            s.trace_focus = false;
                            let idx = s.work_offset + (mouse.row.saturating_sub(inner.y) as usize);
                            if s.work_show_diff {
                                if idx < s.work_diff_files.len() {
                                    s.work_diff_selected = idx;
                                    s.load_work_diff_preview();
                                }
                            } else if idx < s.agent_rows.len() {
                                s.work_selected = idx;
                            }
                            s.mark_dirty();
                            return true;
                        }
                    }
                    if s.files_focus || s.work_focus {
                        if let Some(area) = s.composer_area {
                            if mouse.row >= area.y && mouse.row < area.y.saturating_add(area.height)
                            {
                                s.files_focus = false;
                                s.work_focus = false;
                                s.mark_dirty();
                            }
                        }
                    }
                    if let Some(id) = s
                        .hit_tools
                        .iter()
                        .find(|(h, _)| h.contains(mouse.column, mouse.row))
                        .map(|(_, id)| id.clone())
                    {
                        let is_edit = s.messages.iter().any(|m| {
                            m.id == id
                                && matches!(&m.role, MessageRole::Tool { name, .. } if crate::ui::stream::tool_kind(name) == crate::ui::stream::ToolKind::Edit)
                        });
                        if is_edit {
                            s.toggle_edit_diff(&id);
                        } else {
                            s.toggle_tool_expand(&id);
                        }
                        return true;
                    }
                    if let Some(area) = s.queue_area {
                        match click_queue(&mut s, area, mouse.column, mouse.row) {
                            QueueClick::Edit(i) => {
                                s.queue_edit = Some(i);
                                s.mark_dirty();
                                if let Some(text) = s.prompt_queue.get(i).cloned() {
                                    drop(s);
                                    set_prompt_text(textarea, &text);
                                    return true;
                                }
                            }
                            QueueClick::Send(i) => {
                                s.queue_edit = Some(i);
                                s.mark_dirty();
                                drop(s);
                                let _ = send_now(String::new(), state, client, textarea).await;
                                return true;
                            }
                            QueueClick::Drop(i) => {
                                let was_edit = s.queue_edit == Some(i);
                                s.drop_queued(i);
                                s.set_toast("dropped from queue");
                                drop(s);
                                if was_edit {
                                    *textarea = reset_prompt(false);
                                }
                                return true;
                            }
                            QueueClick::None => {}
                        }
                    }
                    if let Some(area) = s.composer_area {
                        if click_composer(textarea, area, mouse.column, mouse.row) {
                            s.trace_focus = false;
                            let DataCursor(row, col) = textarea.cursor();
                            let lines: Vec<String> =
                                textarea.lines().iter().map(|l| l.to_string()).collect();
                            let off = crate::paste::byte_offset(&lines, row, col);
                            let joined = lines.join("\n");
                            if let Some(tok) = crate::paste::token_at(&joined, off) {
                                if s.open_bracket(tok) {
                                    return true;
                                }
                            }
                        }
                    }
                    PickerAction::None
                } else {
                    PickerAction::None
                }
            };
            apply_picker_action(action, state, client).await;
            true
        }
        _ => false,
    }
}
