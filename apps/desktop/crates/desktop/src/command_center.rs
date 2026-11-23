//! Published v0.23.3 Command Center session list / Agents panel projected onto
//! the native runtime's real local catalog and single interactive session.
//! Sources: app/command-center/index.tsx, app/agents/index.tsx, overlay primitives.
use crate::appearance::text;
use super::{Hermes, Message, Screen, ui};
use hermes_engine::{Entry, Status};
use hermes_library::{SessionSource, SessionSummary};
use iced::{
    Alignment, Border, Color, Element, Fill, Font, Padding,
    widget::{Space, button, column, container, row, scrollable, text_input},
};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum Source {
    #[default]
    All,
    Native,
    Imported,
    Archived,
}
impl Source {
    fn label(self) -> &'static str {
        match self {
            Self::All => "All sources",
            Self::Native => "Local",
            Self::Imported => "Imported",
            Self::Archived => "Archived",
        }
    }
    fn matches(self, session: &SessionSummary) -> bool {
        match self {
            Self::All => true,
            Self::Native => !matches!(session.source, SessionSource::Hermes { .. }),
            Self::Imported => matches!(session.source, SessionSource::Hermes { .. }),
            Self::Archived => session.archived,
        }
    }
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let query = app.command_search.trim().to_lowercase();
    let mut matches = app
        .command_sessions
        .iter()
        .filter(|session| {
            if !app.command_source.matches(session) {
                return false;
            }
            let project = session
                .project_id
                .as_ref()
                .and_then(|id| app.projects.iter().find(|project| &project.id == id))
                .map(|project| project.name.as_str())
                .unwrap_or("");
            let haystack = format!(
                "{} {} {} {}",
                session.title,
                session.id,
                source_label(&session.source),
                project
            )
            .to_lowercase();
            query.is_empty() || haystack.contains(&query)
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        right
            .updated_at
            .cmp(&left.updated_at)
            .then_with(|| left.id.cmp(&right.id))
    });
    let total = matches.len();
    let mut filters = row![].spacing(4);
    for source in [
        Source::All,
        Source::Native,
        Source::Imported,
        Source::Archived,
    ] {
        let selected = source == app.command_source;
        filters = filters.push(
            button(text(source.label()).size(11))
                .on_press(Message::CommandSource(source))
                .padding([5, 9])
                .style(move |_, status| choice_style(p, selected, status)),
        );
    }
    let mut content = column![filters, Space::new().height(12)].spacing(2);
    if let Some(error) = &app.command_error {
        content = content.push(text(error).size(12).color(p.muted));
    }
    if total == 0 {
        content = content.push(
            container(
                text(if query.is_empty() && app.command_source == Source::All {
                    "No sessions yet."
                } else {
                    "No matching results found."
                })
                .size(13)
                .color(p.muted),
            )
            .width(Fill)
            .height(180)
            .center_x(Fill)
            .center_y(180),
        );
    } else {
        for session in matches.into_iter().take(250) {
            content = content.push(session_row(app, session));
        }
        if total > 250 {
            content=content.push(container(text(format!("Showing 250 of {total} matching sessions. Search by title, session ID, project, or source to narrow the list.")).size(11).color(p.muted)).padding([16,0]));
        }
    }
    let search = text_input("Search sessions", &app.command_search)
        .on_input(Message::CommandSearch)
        .size(12)
        .padding([6, 8])
        .width(280)
        .style(move |theme, status| {
            let mut style = text_input::default(theme, status);
            style.background = Color::TRANSPARENT.into();
            style.border.width = 0.0;
            style.value = p.foreground;
            style.placeholder = p.muted;
            style
        });
    let body = column![
        container(
            row![
                search,
                Space::new().width(Fill),
                text(format!("{total} sessions")).size(11).color(p.muted)
            ]
            .align_y(Alignment::Center)
        )
        .padding(Padding {
            top: 0.0,
            right: 32.0,
            bottom: 12.0,
            left: 24.0
        }),
        scrollable(container(content).padding([0, 32]).width(Fill)).height(Fill)
    ]
    .height(Fill);
    shell(app, "Sessions", body.into(), false)
}

pub(super) fn agents(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let state = if app.stopping {
        "Stop requested"
    } else if app.approval.is_some() {
        "Awaiting approval"
    } else if app.running {
        "Running"
    } else {
        "Idle"
    };
    let mut content = column![
        text("Local").size(18).font(medium()),
        text(state)
            .size(12)
            .color(if app.running { p.accent } else { p.muted }),
        Space::new().height(12),
        metadata("Profile", "default".to_owned(), p),
        metadata("Provider", provider_label(app).to_owned(), p),
        metadata(
            "Model",
            app.history
                .turns
                .last()
                .map(|turn| turn.model.clone())
                .filter(|model| !model.is_empty())
                .unwrap_or_else(|| app.config.model.clone()),
            p
        ),
    ]
    .spacing(9);
    if let Some(session) = &app.current {
        content = content
            .push(Space::new().height(16))
            .push(text(&session.title).size(14).font(medium()))
            .push(metadata("Session", session.id.clone(), p))
            .push(metadata(
                "Source",
                source_label(&session.source).to_owned(),
                p,
            ));
        if let Some(project) = app.chat_project() {
            content = content.push(metadata("Project", project.name.clone(), p));
        }
        if let Some(turn) = app.history.turns.last() {
            content = content.push(text(short(super::group_prompt(&turn.prompt).0, 900)).size(12).color(p.muted));
            if let Status::Failed(reason) = &turn.status {
                content = content.push(text(reason).size(12).color(p.muted));
            }
        }
        let mut actions = row![
            button(text("Open session").size(12))
                .on_press(Message::Navigate(Screen::Chat))
                .padding([6, 10])
                .style(button::secondary)
        ]
        .spacing(8);
        if app.running {
            let mut stop = button(
                text(if app.stopping {
                    "Stop requested"
                } else {
                    "Stop"
                })
                .size(12),
            )
            .padding([6, 10])
            .style(button::danger);
            if !app.stopping {
                stop = stop.on_press(Message::Stop);
            }
            actions = actions.push(stop);
        }
        content = content.push(actions);
    } else {
        content = content.push(text("No session selected.").size(12).color(p.muted));
    }
    if let Some(approval) = &app.approval {
        let arguments = serde_json::from_str::<serde_json::Value>(&approval.call.arguments)
            .and_then(|value| serde_json::to_string_pretty(&value))
            .unwrap_or_else(|_| approval.call.arguments.clone());
        content = content
            .push(Space::new().height(18))
            .push(text("Approval required").size(14).font(medium()))
            .push(text(&approval.call.name).size(12).font(Font::MONOSPACE))
            .push(
                scrollable(
                    container(
                        text(arguments)
                            .size(11)
                            .font(Font::MONOSPACE)
                            .color(p.muted),
                    )
                    .padding(10),
                )
                .height(150),
            )
            .push(
                row![
                    button(text("Deny").size(12))
                        .on_press(Message::Approve(false))
                        .padding([6, 10])
                        .style(button::secondary),
                    button(text("Allow once").size(12))
                        .on_press(Message::Approve(true))
                        .padding([6, 10])
                        .style(button::primary)
                ]
                .spacing(8),
            );
    }
    content = content.push(Space::new().height(18)).push(spawn_tree(app));
    let records = app
        .history
        .turns
        .iter()
        .flat_map(|turn| &turn.entries)
        .filter(|entry| matches!(entry, Entry::Tool { .. }))
        .count();
    if records > 0 {
        content = content.push(Space::new().height(12)).push(
            text(format!("{records} tool results recorded in this session"))
                .size(11)
                .color(p.muted),
        );
    }
    shell(
        app,
        "Agents",
        scrollable(container(content).padding([12, 32]).width(Fill))
            .height(Fill)
            .into(),
        true,
    )
}

fn session_row<'a>(app: &'a Hermes, session: &'a SessionSummary) -> Element<'a, Message> {
    let p = ui::palette(app.dark);
    let current = app
        .current
        .as_ref()
        .is_some_and(|active| active.id == session.id);
    let live = current && app.running;
    let state = if live {
        if app.approval.is_some() {
            "Awaiting approval"
        } else if app.stopping {
            "Stop requested"
        } else {
            "Running"
        }
    } else if session.archived {
        "Archived"
    } else if current && app.fresh_draft {
        "Draft"
    } else {
        "Saved"
    };
    let mut details = format!("{} · {}", source_label(&session.source), session.updated_at);
    if let Some(project) = session
        .project_id
        .as_ref()
        .and_then(|id| app.projects.iter().find(|project| &project.id == id))
    {
        details.push_str(&format!(" · {}", project.name));
    }
    let mut open = button(
        column![
            text(short(&session.title, 100)).size(14).font(medium()),
            text(details).size(11).color(p.muted),
            text(&session.id)
                .size(10)
                .font(Font::MONOSPACE)
                .color(p.muted)
        ]
        .spacing(5),
    )
    .padding([9, 0])
    .width(Fill)
    .style(button::text);
    if current {
        open = open.on_press(Message::Navigate(Screen::Chat));
    } else if !app.running {
        open = open.on_press(Message::CommandOpenSession(session.id.clone()));
    }
    let mut controls = row![
        text(state)
            .size(11)
            .color(if live { p.accent } else { p.muted })
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    if live && !app.stopping {
        controls = controls.push(
            button(text("Stop").size(11))
                .on_press(Message::Stop)
                .padding([4, 8])
                .style(button::text),
        );
    }
    if session.pinned {
        controls = controls.push(glyph('\u{eb2b}', 12, p.accent));
    }
    container(row![open, controls].spacing(12).align_y(Alignment::Center))
        .width(Fill)
        .into()
}

fn shell<'a>(
    app: &'a Hermes,
    section: &'static str,
    body: Element<'a, Message>,
    agents: bool,
) -> Element<'a, Message> {
    let p = ui::palette(app.dark);
    let rail = container(
        column![
            button(row![glyph('\u{ea6b}', 14, p.muted), text("Sessions").size(12)].spacing(8))
                .on_press(Message::Navigate(Screen::CommandCenter))
                .padding([8, 10])
                .width(Fill)
                .style(move |_, status| choice_style(p, !agents, status)),
            button(row![glyph('\u{eb08}', 14, p.muted), text("Agents").size(12)].spacing(8))
                .on_press(Message::Navigate(Screen::Agents))
                .padding([8, 10])
                .width(Fill)
                .style(move |_, status| choice_style(p, agents, status)),
            Space::new().height(Fill),
            text("Local").size(11).color(p.muted),
        ]
        .spacing(5),
    )
    .padding([16, 10])
    .width(208)
    .height(Fill)
    .style(move |_| container::Style {
        background: Some(p.sidebar.into()),
        ..Default::default()
    });
    let header = row![
        text("Command Center").size(12).color(p.muted),
        glyph('\u{eab6}', 12, p.muted),
        text(section).size(12),
        Space::new().width(Fill),
        button(glyph('\u{eb37}', 14, p.muted))
            .on_press(Message::CommandRefresh)
            .padding(6)
            .style(button::text),
        button(glyph('\u{ea76}', 16, p.muted))
            .on_press(Message::Navigate(Screen::Chat))
            .padding(6)
            .style(button::text),
    ]
    .spacing(7)
    .align_y(Alignment::Center);
    container(row![
        rail,
        column![
            container(header).padding(Padding {
                top: 8.0,
                right: 12.0,
                bottom: 14.0,
                left: 32.0
            }),
            body
        ]
        .width(Fill)
        .height(Fill)
    ])
    .width(Fill)
    .height(Fill)
    .style(move |_| container::Style {
        background: Some(p.background.into()),
        text_color: Some(p.foreground),
        border: Border {
            color: p.border,
            width: 1.0,
            radius: 12.0.into(),
        },
        ..Default::default()
    })
    .into()
}

fn metadata(label: &'static str, value: String, p: ui::Palette) -> Element<'static, Message> {
    row![
        text(label).size(12).color(p.muted).width(90),
        text(value).size(12).color(p.foreground)
    ]
    .spacing(12)
    .into()
}
fn source_label(source: &SessionSource) -> &'static str {
    match source {
        SessionSource::Native => "Local",
        SessionSource::LegacyRust => "Legacy Rust",
        SessionSource::Hermes { .. } => "Imported Hermes",
    }
}
fn provider_label(app: &Hermes) -> &'static str {
    if app.is_portal() {
        "Nous"
    } else if app.config.base_url.contains("api.anthropic.com") {
        "Anthropic"
    } else if app.config.base_url.contains("openrouter.ai") {
        "OpenRouter"
    } else if app.config.base_url.contains("api.openai.com") {
        "OpenAI"
    } else {
        "Custom endpoint"
    }
}
fn medium() -> Font {
    Font {
        weight: iced::font::Weight::Medium,
        ..Font::DEFAULT
    }
}
fn glyph(value: char, size: u16, color: Color) -> iced::widget::Text<'static> {
    text(value.to_string())
        .font(Font::with_name("codicon"))
        .size(u32::from(size))
        .color(color)
}
fn short(value: &str, limit: usize) -> String {
    let mut chars = value.chars();
    let result = chars.by_ref().take(limit).collect::<String>();
    if chars.next().is_some() {
        format!("{result}…")
    } else {
        result
    }
}
fn choice_style(p: ui::Palette, selected: bool, status: button::Status) -> button::Style {
    button::Style {
        background: (selected || status == button::Status::Hovered)
            .then_some(if selected { p.secondary } else { p.card }.into()),
        text_color: if selected { p.accent } else { p.foreground },
        border: Border {
            radius: 4.0.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Agents › Spawn tree (app/agents): live subagent activity for the current turn,
/// grouped by delegation, with each child's state, tool use and answer.
fn spawn_tree(app: &Hermes) -> Element<'_, Message> {
    let p = ui::palette(app.dark);
    let list = &app.subagents;
    let mut col = column![text("Spawn tree").size(14).font(medium()), text("Live subagent activity for the current turn.").size(12).color(p.muted)].spacing(6);
    if list.is_empty() {
        return col.push(container(column![ui::icon("hubot", 20, p.muted), text("No live subagents").size(13), text("When a turn delegates work, child agents stream their progress here.").size(12).color(p.muted)]
            .spacing(5).align_x(Alignment::Center)).padding(24).center_x(Fill)).into();
    }
    let active = list.iter().filter(|s| s.status == "running" || s.status == "queued").count();
    let failed = list.iter().filter(|s| s.status == "failed").count();
    col = col.push(text(format!("{} agents{}{}", list.len(), if active > 0 { format!(" · {active} active") } else { String::new() }, if failed > 0 { format!(" · {failed} failed") } else { String::new() })).size(11).color(p.muted));
    let mut groups: Vec<&str> = list.iter().map(|s| s.delegation_id.as_str()).collect();
    groups.dedup();
    let now = chrono::Utc::now().timestamp();
    for (n, group) in groups.iter().enumerate() {
        col = col.push(text(format!("DELEGATION {}", n + 1)).size(10).font(medium()).color(p.muted));
        for agent in list.iter().filter(|s| s.delegation_id == *group) {
            let (glyph, color) = match agent.status.as_str() {
                "completed" => ("check", Color::from_rgb8(0x1f, 0x8a, 0x65)),
                "failed" => ("error", Color::from_rgb8(0xd1, 0x43, 0x43)),
                "interrupted" => ("debug-pause", p.muted),
                "queued" => ("watch", p.muted),
                _ => ("loading", p.accent),
            };
            let elapsed = (if matches!(agent.status.as_str(), "running" | "queued") { now } else { agent.updated_at }) - agent.started_at;
            let mut meta = vec![format!("{}s", elapsed.max(0))];
            if agent.tool_count > 0 { meta.push(format!("{} tool call{}", agent.tool_count, if agent.tool_count == 1 { "" } else { "s" })); }
            if let Some(tool) = &agent.current_tool { meta.push(format!("running {tool}")); }
            let open = app.subagent_open.as_deref() == Some(agent.id.as_str());
            let mut card = column![
                row![ui::icon(glyph, 13, color), text(short(&agent.goal, 140)).size(13).width(Fill),
                    text(format!("{}/{}", agent.task_index + 1, agent.task_count)).size(11).color(p.muted),
                    button(text(if open { "Hide" } else { "Details" }).size(11)).padding([2, 6]).style(button::text)
                        .on_press(Message::SubagentOpen(if open { None } else { Some(agent.id.clone()) }))].spacing(8).align_y(Alignment::Center),
                text(meta.join(" · ")).size(11).color(p.muted),
            ].spacing(4);
            if !agent.preview.is_empty() && agent.status == "running" { card = card.push(text(short(&agent.preview, 220)).size(11).color(p.muted)); }
            if open {
                if let Some(summary) = &agent.summary { card = card.push(container(text(summary.clone()).size(12)).padding(8).width(Fill).style(move |_| ui::panel(p.secondary, 4.0, None))); }
                if let Ok(history) = std::fs::read(agent.session_dir.join("conversation.json")).map_err(|_| ()).and_then(|b| serde_json::from_slice::<hermes_engine::History>(&b).map_err(|_| ())) {
                    for entry in history.turns.iter().flat_map(|t| &t.entries) {
                        let line = match entry {
                            Entry::Assistant { calls, .. } if !calls.is_empty() => calls.iter().map(|c| format!("→ {}", c.name)).collect::<Vec<_>>().join("  "),
                            Entry::Tool { name, approved, .. } => format!("← {name}{}", if *approved { "" } else { " (denied)" }),
                            _ => continue,
                        };
                        card = card.push(text(line).size(11).font(Font::MONOSPACE).color(p.muted));
                    }
                }
            }
            col = col.push(container(card).padding([8, 10]).width(Fill).style(move |_| ui::panel(p.card, 6.0, Some(p.border))));
        }
    }
    col.into()
}
