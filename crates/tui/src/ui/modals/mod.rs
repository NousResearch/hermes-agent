use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    Frame,
};

use crate::state::{ActiveView, AppState};

mod overlays;
mod prompts;

pub struct ViewsOverlay;

impl ViewsOverlay {
    pub fn render(frame: &mut Frame, area: Rect, state: &mut AppState, frame_count: u64) {
        match state.active_view {
            ActiveView::Chat => {
                if state.pending_secret.is_some() {
                    Self::render_secret_modal(frame, area, state);
                } else if state.pending_clarify.is_some() {
                    Self::render_clarify_modal(frame, area, state);
                } else if state.pending_approval.is_some() {
                    Self::render_approval_modal(frame, area, state);
                }
            }
            ActiveView::Tasks => Self::render_tasks_modal(frame, area, state, frame_count),
            ActiveView::ThemePicker => Self::render_theme_modal(frame, area, state),
            ActiveView::Context => crate::ui::context::ContextModal::render(frame, area, state),
            ActiveView::ModelPicker => Self::render_model_modal(frame, area, state),
            ActiveView::BranchPicker => Self::render_branch_modal(frame, area, state),
            ActiveView::Skills => Self::render_skills_modal(frame, area, state),
            ActiveView::Sessions => Self::render_sessions_modal(frame, area, state),
            ActiveView::Profiles => Self::render_profiles_modal(frame, area, state),
            ActiveView::Agents => Self::render_agents_modal(frame, area, state, frame_count),
            ActiveView::Memory => Self::render_memory_modal(frame, area, state),
            ActiveView::Help => Self::render_help_modal(frame, area),
            ActiveView::Peek => Self::render_peek_modal(frame, area, state),
            ActiveView::Rollback => Self::render_rollback_modal(frame, area, state),
            ActiveView::Background => {
                Self::render_background_modal(frame, area, state, frame_count)
            }
            ActiveView::Mcp => Self::render_mcp_modal(frame, area, state),
            ActiveView::Palette => Self::render_palette_modal(frame, area, state),
            ActiveView::Tools => Self::render_tools_modal(frame, area, state),
            ActiveView::Plugins => Self::render_plugins_modal(frame, area, state),
            ActiveView::Cron => Self::render_cron_modal(frame, area, state),
            ActiveView::Replay => Self::render_replay_modal(frame, area, state),
            ActiveView::Projects => Self::render_projects_modal(frame, area, state),
        }
    }

    pub(super) fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
        let popup_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage((100 - percent_y) / 2),
                Constraint::Percentage(percent_y),
                Constraint::Percentage((100 - percent_y) / 2),
            ])
            .split(r);

        Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage((100 - percent_x) / 2),
                Constraint::Percentage(percent_x),
                Constraint::Percentage((100 - percent_x) / 2),
            ])
            .split(popup_layout[1])[1]
    }

    pub(super) fn wrap_detail(text: &str, width: usize) -> Vec<String> {
        crate::layout::wrap_chunks(text, width.max(12))
    }
}
