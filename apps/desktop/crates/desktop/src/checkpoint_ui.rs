//! Manual rollback surface attached to stock Settings → Safety.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use hermes_engine::{
    checkpoint::{RestorePlan, Store, Summary},
    settings::ProfileSettings,
};
use iced::{
    Alignment, Element, Fill, Font, Task,
    widget::{Space, button, column, container, row, scrollable},
};
use std::path::PathBuf;
#[derive(Clone, Debug)]
pub enum Action {
    Refresh,
    Review(String),
    Cancel,
    Confirm,
    Loaded(u64, Result<Vec<Summary>, String>),
    Planned(u64, Result<RestorePlan, String>),
    Restored(u64, Result<Summary, String>),
}
#[derive(Default)]
pub struct State {
    profile: Option<PathBuf>,
    records: Vec<Summary>,
    plan: Option<RestorePlan>,
    feedback: Option<Result<String, String>>,
    busy: bool,
    request: u64,
}
fn store(app: &Hermes) -> Result<Store, String> {
    let root = app
        .library
        .as_ref()
        .ok_or("Open a local profile first")?
        .root();
    let settings = ProfileSettings::load(root).map_err(|error| error.to_string())?;
    Store::open(root, settings.checkpoint_max_snapshots).map_err(|error| format!("{error:#}"))
}
pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    let profile = app
        .library
        .as_ref()
        .map(|library| library.root().to_path_buf());
    if app.checkpoints.profile != profile {
        let request = app.checkpoints.request.wrapping_add(1);
        app.checkpoints = State {
            profile,
            request,
            ..State::default()
        };
    }
    match action {
        Action::Refresh if !app.checkpoints.busy => {
            let store = match store(app) {
                Ok(store) => store,
                Err(error) => {
                    app.checkpoints.feedback = Some(Err(error));
                    return Task::none();
                }
            };
            app.checkpoints.request = app.checkpoints.request.wrapping_add(1);
            let request = app.checkpoints.request;
            app.checkpoints.busy = true;
            app.checkpoints.plan = None;
            return Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || {
                        store.list().map_err(|error| format!("{error:#}"))
                    })
                    .await
                    .unwrap_or_else(|error| Err(error.to_string()))
                },
                move |result| Message::Checkpoint(Action::Loaded(request, result)),
            );
        }
        Action::Review(id) => {
            let store = match store(app) {
                Ok(store) => store,
                Err(error) => {
                    app.checkpoints.feedback = Some(Err(error));
                    return Task::none();
                }
            };
            app.checkpoints.request = app.checkpoints.request.wrapping_add(1);
            let request = app.checkpoints.request;
            app.checkpoints.busy = true;
            app.checkpoints.feedback = None;
            return Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || {
                        store
                            .prepare_restore(&id)
                            .map_err(|error| format!("{error:#}"))
                    })
                    .await
                    .unwrap_or_else(|error| Err(error.to_string()))
                },
                move |result| Message::Checkpoint(Action::Planned(request, result)),
            );
        }
        Action::Cancel => {
            app.checkpoints.plan = None;
            app.checkpoints.feedback = None;
        }
        Action::Confirm if !app.checkpoints.busy => {
            let Some(plan) = app.checkpoints.plan.clone() else {
                return Task::none();
            };
            let store = match store(app) {
                Ok(store) => store,
                Err(error) => {
                    app.checkpoints.feedback = Some(Err(error));
                    return Task::none();
                }
            };
            app.checkpoints.request = app.checkpoints.request.wrapping_add(1);
            let request = app.checkpoints.request;
            app.checkpoints.busy = true;
            return Task::perform(
                async move {
                    tokio::task::spawn_blocking(move || {
                        store
                            .restore(&plan, true)
                            .map_err(|error| format!("{error:#}"))
                    })
                    .await
                    .unwrap_or_else(|error| Err(error.to_string()))
                },
                move |result| Message::Checkpoint(Action::Restored(request, result)),
            );
        }
        Action::Loaded(request, result) if request == app.checkpoints.request => {
            app.checkpoints.busy = false;
            match result {
                Ok(records) => app.checkpoints.records = records,
                Err(error) => app.checkpoints.feedback = Some(Err(error)),
            }
        }
        Action::Planned(request, result) if request == app.checkpoints.request => {
            app.checkpoints.busy = false;
            match result {
                Ok(plan) => app.checkpoints.plan = Some(plan),
                Err(error) => app.checkpoints.feedback = Some(Err(error)),
            }
        }
        Action::Restored(request, result) if request == app.checkpoints.request => {
            app.checkpoints.busy = false;
            app.checkpoints.plan = None;
            match result {
                Ok(summary) => {
                    app.checkpoints.feedback = Some(Ok(format!(
                        "Restored {}. A pre-rollback checkpoint was saved.",
                        summary.path.display()
                    )));
                    return update(app, Action::Refresh);
                }
                Err(error) => app.checkpoints.feedback = Some(Err(error)),
            }
        }
        _ => {}
    }
    Task::none()
}
pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let state = &app.checkpoints;
    let mut body=column![row![text("Filesystem checkpoints").size(14).width(Fill),button(text(if state.busy{"Loading..."}else{"Refresh checkpoints"}).size(11)).on_press_maybe((!state.busy).then_some(Message::Checkpoint(Action::Refresh))).style(button::text)].spacing(10).align_y(Alignment::Center),
        text("Snapshots cover approved native file writes when checkpoints are enabled. Restore refuses files changed after the saved write. Your project's Git repository is unchanged.").size(11).color(p.muted)].spacing(10).width(Fill);
    if let Some(feedback) = &state.feedback {
        body = body.push(
            text(match feedback {
                Ok(message) | Err(message) => message,
            })
            .size(12)
            .color(if feedback.is_err() {
                if app.dark {
                    iced::Color::from_rgb8(248, 81, 73)
                } else {
                    iced::Color::from_rgb8(207, 34, 46)
                }
            } else {
                p.accent
            }),
        );
    }
    if let Some(plan) = &state.plan {
        body = body
            .push(text("Restore to this checkpoint?").size(14))
            .push(text(&plan.description).size(12))
            .push(
                text(format!(
                    "Workspace: {}\nFile: {}",
                    plan.workspace.display(),
                    plan.path.display()
                ))
                .size(11)
                .color(p.muted),
            )
            .push(scrollable(text(&plan.preview).size(11).font(Font::MONOSPACE)).height(170))
            .push(
                row![
                    Space::new().width(Fill),
                    button(text("Cancel").size(12))
                        .on_press_maybe(
                            (!state.busy).then_some(Message::Checkpoint(Action::Cancel))
                        )
                        .style(button::text),
                    button(text("Restore checkpoint").size(12))
                        .on_press_maybe(
                            (!state.busy).then_some(Message::Checkpoint(Action::Confirm))
                        )
                        .style(button::danger)
                ]
                .spacing(8),
            );
    } else if state.records.is_empty() {
        body = body.push(
            text("No filesystem checkpoints yet.")
                .size(12)
                .color(p.muted),
        );
    } else {
        let mut records = column![].spacing(6);
        for record in &state.records {
            let action: Element<'_, Message> = if record.restored {
                text("Restored").size(11).color(p.muted).into()
            } else if !record.ready {
                text("Write not completed").size(11).color(p.muted).into()
            } else {
                button(text("Review restore").size(11))
                    .on_press_maybe(
                        (!state.busy)
                            .then(|| Message::Checkpoint(Action::Review(record.id.clone()))),
                    )
                    .style(button::text)
                    .into()
            };
            records = records.push(
                row![
                    column![
                        text(record.path.to_string_lossy().into_owned()).size(12),
                        text(format!(
                            "{} · {} · {} → {} bytes",
                            record.created_at,
                            record.label,
                            record.before_bytes,
                            record.after_bytes
                        ))
                        .size(10)
                        .color(p.muted),
                        text(record.workspace.to_string_lossy().into_owned())
                            .size(10)
                            .color(p.muted)
                    ]
                    .spacing(3)
                    .width(Fill),
                    action
                ]
                .spacing(8)
                .align_y(Alignment::Center),
            );
        }
        body = body.push(scrollable(records).height(220));
    }
    container(body).padding([14, 0]).width(Fill).into()
}
