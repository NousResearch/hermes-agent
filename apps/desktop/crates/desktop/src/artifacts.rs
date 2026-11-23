//! Native Artifacts route, based on official v0.23.3 (caafe5389).
//! Source: `apps/desktop/src/app/artifacts/` and PageSearchShell, read from
//! the immutable public-release Git tree. artifact-utils blob: 50b26eee.
//! Indexing reads conversation records only. Local file reads and browser
//! navigation happen exclusively in response to the corresponding user action.
mod model;

use crate::appearance::text;
use base64::Engine;
use chrono::{Local, TimeZone};
use hermes_engine::History;
use hermes_library::{Library, SessionSummary};
use iced::{
    Alignment, Border, Color, Element, Fill, Font, Length, Padding, Task,
    widget::{
        Space, button, column, container, image, responsive, row, rule, scrollable, stack,
        text_input, tooltip,
    },
};
use model::{Artifact, Filter, Index, Kind};
use std::{
    fmt,
    fs::File,
    io::{Cursor, Read},
    sync::Arc,
};

const IMAGE_PAGE_SIZE: usize = 24;
const FILE_PAGE_SIZE: usize = 100;
const TEXT_PREVIEW_BYTES: usize = 96 * 1024;
const IMAGE_PREVIEW_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone)]
pub(super) enum Message {
    Refresh,
    Loaded(u64, Result<Arc<Index>, String>),
    Search(String),
    Filter(Filter),
    ImagePage(usize),
    FilePage(usize),
    Open(String),
    Previewed(u64, String, Result<Arc<Preview>, String>),
    OpenSession(String),
    Copy(String),
    ClosePreview,
    /// Open the previewed HTML file in the OS browser (a real click on the header button).
    OpenInBrowser,
    /// Open the link the guest page tried to follow (offered in the header).
    OpenOffered(String),
}
impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Refresh => "RefreshArtifacts",
            Self::Loaded(..) => "ArtifactsLoaded",
            Self::Search(_) => "SearchArtifacts",
            Self::Filter(_) => "FilterArtifacts",
            Self::ImagePage(_) | Self::FilePage(_) => "ArtifactPage",
            Self::Open(_) => "OpenArtifact",
            Self::Previewed(..) => "ArtifactPreviewed",
            Self::OpenSession(_) => "OpenArtifactSession",
            Self::Copy(_) => "CopyArtifactLocation",
            Self::ClosePreview => "CloseArtifactPreview", Self::OpenInBrowser => "OpenArtifactInBrowser", Self::OpenOffered(_) => "OpenOfferedLink",
        })
    }
}

pub(super) enum Effect {
    OpenSession(String),
    OpenLink(String),
    OpenFile(std::path::PathBuf),
    Copy(String),
}

pub(super) enum Preview {
    Text {
        text: String,
        bytes: u64,
        truncated: bool,
    },
    Image {
        handle: image::Handle,
        width: u32,
        height: u32,
    },
    Binary {
        bytes: u64,
    },
    /// An HTML document shown in the native webview (webview.rs), scoped to its folder.
    Html {
        path: std::path::PathBuf,
        bytes: u64,
    },
}

#[derive(Default)]
pub(super) struct Artifacts {
    index: Option<Arc<Index>>,
    query: String,
    filter: Filter,
    image_page: usize,
    file_page: usize,
    refreshing: bool,
    error: Option<String>,
    generation: u64,
    preview_generation: u64,
    preview_id: Option<String>,
    preview: Option<Arc<Preview>>,
    preview_loading: bool,
}

impl Artifacts {
    pub(super) fn refresh(
        &mut self,
        library: Option<Library>,
        live: Option<(SessionSummary, History)>,
    ) -> Task<Message> {
        if self.refreshing {
            return Task::none();
        }
        let Some(library) = library else {
            self.error = Some("Artifacts failed to load: chat library is unavailable.".to_owned());
            return Task::none();
        };
        self.refreshing = true;
        self.error = None;
        self.generation = self.generation.wrapping_add(1);
        let generation = self.generation;
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || model::load(&library, live))
                    .await
                    .map_err(|error| error.to_string())
                    .and_then(|result| result)
                    .map(Arc::new)
            },
            move |result| Message::Loaded(generation, result),
        )
    }

    pub(super) fn update(
        &mut self,
        message: Message,
        library: Option<Library>,
        live: Option<(SessionSummary, History)>,
    ) -> (Task<Message>, Option<Effect>) {
        match message {
            Message::Refresh => return (self.refresh(library, live), None),
            Message::Loaded(generation, result) if generation == self.generation => {
                self.refreshing = false;
                self.image_page = 1;
                self.file_page = 1;
                match result {
                    Ok(index) => {
                        // Test-only (ui-test builds): open the first HTML artifact to verify the native preview.
                        #[cfg(feature = "ui-test")]
                        let first_html = std::env::var_os("HERMES_RUST_OPEN_ARTIFACT").and_then(|_| index.artifacts.iter()
                            .find(|a| a.value.to_ascii_lowercase().ends_with(".html")).map(|a| a.id.clone()));
                        self.index = Some(index);
                        #[cfg(feature = "ui-test")]
                        if let Some(id) = first_html { return self.update(Message::Open(id), library, live); }
                    }
                    Err(error) => self.error = Some(format!("Artifacts failed to load: {error}")),
                }
            }
            Message::Search(query) => {
                self.query = query;
                self.image_page = 1;
                self.file_page = 1;
            }
            Message::Filter(filter) => {
                self.filter = filter;
                self.image_page = 1;
                self.file_page = 1;
            }
            Message::ImagePage(page) => self.image_page = page.max(1),
            Message::FilePage(page) => self.file_page = page.max(1),
            Message::Open(id) => {
                let Some(artifact) = self.artifact(&id).cloned() else {
                    return (Task::none(), None);
                };
                if artifact.value.starts_with("https://") || artifact.value.starts_with("http://") {
                    match url::Url::parse(&artifact.value) {
                        Ok(url) if matches!(url.scheme(), "http" | "https") => {
                            return (Task::none(), Some(Effect::OpenLink(url.to_string())));
                        }
                        _ => {
                            self.error = Some("Open failed: invalid web address.".into());
                            return (Task::none(), None);
                        }
                    }
                }
                self.preview_generation = self.preview_generation.wrapping_add(1);
                let generation = self.preview_generation;
                self.preview_id = Some(id.clone());
                self.preview = None;
                self.preview_loading = true;
                self.error = None;
                return (
                    Task::perform(
                        async move {
                            tokio::task::spawn_blocking(move || preview(&artifact))
                                .await
                                .map_err(|error| error.to_string())
                                .and_then(|result| result)
                                .map(Arc::new)
                        },
                        move |result| Message::Previewed(generation, id.clone(), result),
                    ),
                    None,
                );
            }
            Message::Previewed(generation, id, result)
                if generation == self.preview_generation
                    && self.preview_id.as_deref() == Some(&id) =>
            {
                self.preview_loading = false;
                match result {
                    Ok(preview) => self.preview = Some(preview),
                    Err(error) => {
                        self.preview_id = None;
                        self.error = Some(format!("Open failed: {error}"));
                    }
                }
            }
            Message::OpenSession(id) => {
                if self.index.as_ref().is_some_and(|index| {
                    index
                        .artifacts
                        .iter()
                        .any(|artifact| artifact.session_id == id)
                }) {
                    self.close_preview();
                    return (Task::none(), Some(Effect::OpenSession(id)));
                }
            }
            Message::Copy(id) => {
                if let Some(artifact) = self.artifact(&id) {
                    return (Task::none(), Some(Effect::Copy(artifact.value.clone())));
                }
            }
            Message::ClosePreview => self.close_preview(),
            Message::OpenInBrowser => if let Some(path) = self.html_preview() { return (Task::none(), Some(Effect::OpenFile(path))); },
            Message::OpenOffered(url) => return (Task::none(), Some(Effect::OpenLink(url))),
            _ => {}
        }
        (Task::none(), None)
    }

    fn close_preview(&mut self) {
        self.preview_generation = self.preview_generation.wrapping_add(1);
        self.preview = None;
        self.preview_id = None;
        self.preview_loading = false;
    }
    fn artifact(&self, id: &str) -> Option<&Artifact> {
        self.index
            .as_ref()?
            .artifacts
            .iter()
            .find(|artifact| artifact.id == id)
    }

    #[cfg(debug_assertions)]
    pub(super) fn trace(&self) -> String {
        format!("id={:?} kind={}", self.preview_id, match self.preview.as_deref() { Some(Preview::Html { .. }) => "html", Some(Preview::Text { .. }) => "text", Some(Preview::Image { .. }) => "image", Some(Preview::Binary { .. }) => "binary", None => "none" })
    }
    /// The HTML file the preview pane is showing, if any.
    pub(super) fn html_preview(&self) -> Option<std::path::PathBuf> {
        match self.preview.as_deref() { Some(Preview::Html { path, .. }) if self.preview_id.is_some() => Some(path.clone()), _ => None }
    }
    pub(super) fn view(&self, dark: bool, running: bool, offered: Option<String>) -> Element<'_, Message> {
        responsive(move |size| {
            self.page(
                dark,
                running,
                ((size.width - 24.0) / 184.0).floor().max(1.0) as usize,
                offered.clone(),
            )
        })
        .into()
    }

    fn page(&self, dark: bool, running: bool, columns: usize, offered: Option<String>) -> Element<'_, Message> {
        let p = super::ui::palette(dark);
        let all = self
            .index
            .as_ref()
            .map(|index| index.artifacts.as_slice())
            .unwrap_or_default();
        let mut tabs = row![].spacing(10).align_y(Alignment::Center);
        for filter in Filter::ALL {
            let count = all
                .iter()
                .filter(|artifact| filter.accepts(artifact.kind))
                .count();
            let active = filter == self.filter;
            let tab = column![
                row![
                    text(filter.label()).size(13).font(if active {
                        semibold()
                    } else {
                        crate::appearance::font()
                    }),
                    text(if self.index.is_some() {
                        count.to_string()
                    } else {
                        "–".into()
                    })
                    .size(11)
                    .color(p.muted)
                ]
                .spacing(6),
                container(Space::new().height(2))
                    .width(Fill)
                    .style(move |_| container::Style {
                        background: active.then_some(p.accent.into()),
                        ..Default::default()
                    }),
            ]
            .spacing(7);
            tabs = tabs.push(
                button(tab)
                    .on_press(Message::Filter(filter))
                    .padding([3, 0])
                    .style(button::text),
            );
        }
        let search: Element<'_, _> = if !all.is_empty() {
            row![
                super::ui::icon("search", 14, p.muted),
                text_input("Search artifacts...", &self.query)
                    .on_input(Message::Search)
                    .size(13)
                    .padding([5, 0])
                    .style(move |theme, status| {
                        let mut style = text_input::default(theme, status);
                        style.background = Color::TRANSPARENT.into();
                        style.border.width = 0.0;
                        style.value = p.foreground;
                        style.placeholder = p.muted;
                        style
                    })
            ]
            .spacing(7)
            .align_y(Alignment::Center)
            .into()
        } else {
            Space::new().into()
        };
        let refresh = tooltip(
            button(super::ui::icon(
                if self.refreshing {
                    "loading"
                } else {
                    "refresh"
                },
                16,
                p.muted,
            ))
            .on_press_maybe((!self.refreshing).then_some(Message::Refresh))
            .padding(5)
            .style(button::text),
            text(if self.refreshing {
                "Refreshing artifacts"
            } else {
                "Refresh artifacts"
            })
            .size(11),
            tooltip::Position::Bottom,
        );
        let header = row![
            container(search).width(Length::FillPortion(1)),
            tabs,
            container(refresh)
                .width(Length::FillPortion(1))
                .align_right(Fill)
        ]
        .spacing(12)
        .align_y(Alignment::Center);
        let mut page = column![container(header).padding(Padding {
            top: 8.0,
            right: 12.0,
            bottom: 8.0,
            left: 12.0
        })]
        .height(Fill);
        if let Some(error) = &self.error {
            page = page.push(
                container(text(error).size(12).style(|theme: &iced::Theme| {
                    iced::widget::text::Style {
                        color: Some(theme.palette().danger),
                    }
                }))
                .padding([6, 12]),
            );
        }
        if let Some(index) = &self.index
            && !index.failures.is_empty()
        {
            page = page.push(container(text(format!("Artifacts failed to load · Skipped {} of {} recent sessions while indexing artifacts.",index.failures.len(),index.sessions)).size(12).color(p.muted)).padding([6,12]));
        }
        let visible: Vec<_> = all
            .iter()
            .filter(|artifact| artifact.matches(self.filter, &self.query))
            .collect();
        if self.index.is_none() && self.refreshing {
            page = page.push(
                container(
                    text("Indexing recent session artifacts")
                        .size(13)
                        .color(p.muted),
                )
                .center_x(Fill)
                .center_y(Fill)
                .height(Fill),
            );
        } else if visible.is_empty() {
            page = page.push(container(column![text("No artifacts found").size(14).font(semibold()),text("Generated images and file outputs will appear here as sessions produce them.").size(12).color(p.muted)].spacing(4).align_x(Alignment::Center)).center_x(Fill).center_y(Fill).height(Fill));
        } else {
            let images: Vec<_> = visible
                .iter()
                .copied()
                .filter(|artifact| artifact.kind == Kind::Image)
                .collect();
            let files: Vec<_> = visible
                .iter()
                .copied()
                .filter(|artifact| artifact.kind != Kind::Image)
                .collect();
            let mut content = column![].spacing(12);
            if !images.is_empty() {
                let image_page = current_page(self.image_page, images.len(), IMAGE_PAGE_SIZE);
                content = content.push(pagination(
                    images.len(),
                    image_page,
                    IMAGE_PAGE_SIZE,
                    true,
                    dark,
                ));
                let page: Vec<_> = images
                    .into_iter()
                    .skip((image_page - 1) * IMAGE_PAGE_SIZE)
                    .take(IMAGE_PAGE_SIZE)
                    .collect();
                for chunk in page.chunks(columns) {
                    let mut cards = row![].spacing(8);
                    for artifact in chunk {
                        cards = cards.push(self.image_card(artifact, dark, running));
                    }
                    for _ in chunk.len()..columns {
                        cards = cards.push(Space::new().width(Length::FillPortion(1)));
                    }
                    content = content.push(cards);
                }
            }
            if !files.is_empty() {
                let file_page = current_page(self.file_page, files.len(), FILE_PAGE_SIZE);
                content = content.push(pagination(
                    files.len(),
                    file_page,
                    FILE_PAGE_SIZE,
                    false,
                    dark,
                ));
                let widths = if self.filter == Filter::Links {
                    [50, 30, 20]
                } else {
                    [35, 41, 24]
                };
                let headings = match self.filter {
                    Filter::Links => ["LINK TITLE", "URL", "SESSION"],
                    Filter::Files => ["NAME", "PATH", "SESSION"],
                    _ => ["TITLE / NAME", "LOCATION", "SESSION"],
                };
                let head = row![
                    container(text(headings[0]).size(10).color(p.muted))
                        .width(Length::FillPortion(widths[0])),
                    container(text(headings[1]).size(10).color(p.muted))
                        .width(Length::FillPortion(widths[1])),
                    container(text(headings[2]).size(10).color(p.muted))
                        .width(Length::FillPortion(widths[2]))
                ]
                .spacing(10)
                .padding([7, 10]);
                let mut table = column![
                    container(head)
                        .width(Fill)
                        .style(move |_| container::Style {
                            background: Some(p.card.into()),
                            ..Default::default()
                        }),
                    rule::horizontal(1)
                ]
                .spacing(0);
                for artifact in files
                    .into_iter()
                    .skip((file_page - 1) * FILE_PAGE_SIZE)
                    .take(FILE_PAGE_SIZE)
                {
                    let title = button(
                        row![
                            super::ui::icon(
                                if artifact.kind == Kind::Link {
                                    "link"
                                } else {
                                    "file"
                                },
                                14,
                                p.muted
                            ),
                            text(short(&artifact.label, 70)).size(12)
                        ]
                        .spacing(8)
                        .align_y(Alignment::Center),
                    )
                    .on_press(Message::Open(artifact.id.clone()))
                    .padding([7, 10])
                    .width(Fill)
                    .style(button::text);
                    let location = row![
                        text(short(&artifact.value, 90))
                            .size(12)
                            .color(p.muted)
                            .font(if artifact.kind == Kind::Link {
                                crate::appearance::font()
                            } else {
                                Font::MONOSPACE
                            })
                            .width(Fill),
                        tooltip(
                            button(super::ui::icon("copy", 13, p.muted))
                                .on_press(Message::Copy(artifact.id.clone()))
                                .padding(3)
                                .style(button::text),
                            text(if artifact.kind == Kind::Link {
                                "Copy URL"
                            } else {
                                "Copy path"
                            })
                            .size(11),
                            tooltip::Position::Bottom
                        )
                    ]
                    .spacing(4)
                    .align_y(Alignment::Center);
                    let session = button(
                        column![
                            text(short(&artifact.session_title, 45)).size(12),
                            text(time_label(artifact.timestamp)).size(11).color(p.muted)
                        ]
                        .spacing(3),
                    )
                    .on_press_maybe(
                        (!running).then(|| Message::OpenSession(artifact.session_id.clone())),
                    )
                    .padding([7, 10])
                    .width(Fill)
                    .style(button::text);
                    table = table.push(
                        row![
                            container(title).width(Length::FillPortion(widths[0])),
                            container(location)
                                .padding([0, 10])
                                .width(Length::FillPortion(widths[1])),
                            container(session).width(Length::FillPortion(widths[2]))
                        ]
                        .spacing(0)
                        .align_y(Alignment::Center),
                    );
                }
                content =
                    content.push(
                        container(table)
                            .width(Fill)
                            .style(move |_| container::Style {
                                border: Border {
                                    color: p.border,
                                    width: 1.0,
                                    radius: 8.0.into(),
                                },
                                ..Default::default()
                            }),
                    );
            }
            page = page.push(scrollable(container(content).padding([0, 12])).height(Fill));
        }
        let base: Element<'_, _> = page.into();
        if let Some(id) = &self.preview_id
            && let Some(artifact) = self.artifact(id)
        {
            let preview = self.preview_view(artifact, dark, running, offered.clone());
            stack![
                base,
                container(Space::new().width(Fill).height(Fill)).style(|_| container::Style {
                    background: Some(
                        Color {
                            a: 0.22,
                            ..Color::BLACK
                        }
                        .into()
                    ),
                    ..Default::default()
                }),
                container(preview)
                    .width(Fill)
                    .height(Fill)
                    .padding(24)
                    .center_x(Fill)
                    .center_y(Fill)
            ]
            .into()
        } else {
            base
        }
    }

    fn image_card<'a>(
        &'a self,
        artifact: &'a Artifact,
        dark: bool,
        running: bool,
    ) -> Element<'a, Message> {
        let p = super::ui::palette(dark);
        let preview = button(
            container(
                column![
                    super::ui::icon("file", 28, p.muted),
                    text("Preview").size(12).color(p.muted)
                ]
                .spacing(8)
                .align_x(Alignment::Center),
            )
            .width(Fill)
            .height(160)
            .center_x(Fill)
            .center_y(Fill),
        )
        .on_press(Message::Open(artifact.id.clone()))
        .padding(0)
        .width(Fill)
        .style(button::text);
        container(
            column![
                preview,
                rule::horizontal(1),
                container(
                    column![
                        text("IMAGE").size(10).color(p.muted),
                        text(short(&artifact.label, 38)).size(12).font(semibold()),
                        text(short(&artifact.value, 48)).size(10).color(p.muted),
                        text(format!(
                            "{} · {}",
                            short(&artifact.session_title, 26),
                            time_label(artifact.timestamp)
                        ))
                        .size(10)
                        .color(p.muted),
                        button(
                            row![
                                super::ui::icon("folder-opened", 12, p.muted),
                                text("Chat").size(11)
                            ]
                            .spacing(4)
                        )
                        .on_press_maybe(
                            (!running).then(|| Message::OpenSession(artifact.session_id.clone()))
                        )
                        .padding([2, 0])
                        .style(button::text)
                    ]
                    .spacing(5)
                )
                .padding(8)
            ]
            .spacing(0),
        )
        .width(Length::FillPortion(1))
        .style(move |_| container::Style {
            border: Border {
                color: p.border,
                width: 1.0,
                radius: 8.0.into(),
            },
            ..Default::default()
        })
        .into()
    }

    fn preview_view<'a>(
        &'a self,
        artifact: &'a Artifact,
        dark: bool,
        running: bool,
        offered: Option<String>,
    ) -> Element<'a, Message> {
        let p = super::ui::palette(dark);
        let html = matches!(self.preview.as_deref(), Some(Preview::Html { .. }));
        let mut content = column![
            row![
                text(&artifact.label).size(15).font(semibold()),
                Space::new().width(Fill),
            ]
            .push(if html { Element::from(button(row![text("Open in browser").size(12), super::ui::icon("link-external", 11, p.muted)].spacing(5).align_y(Alignment::Center))
                .on_press(Message::OpenInBrowser).padding([4, 8]).style(button::text)) } else { Space::new().into() })
            .push(
                button(super::ui::icon("close", 16, p.muted))
                    .on_press(Message::ClosePreview)
                    .padding(5)
                    .style(button::text))
            .align_y(Alignment::Center),
            text(short(&artifact.value, 180)).size(12).color(p.muted)
        ]
        .spacing(12)
        .height(Fill);
        let body: Element<'_, _> = match self.preview.as_deref() {
            Some(Preview::Image {
                handle,
                width,
                height,
            }) => column![
                image(handle.clone())
                    .content_fit(iced::ContentFit::Contain)
                    .width(Fill)
                    .height(Fill),
                text(format!("{width} × {height}")).size(11).color(p.muted)
            ]
            .spacing(8)
            .height(Fill)
            .into(),
            Some(Preview::Text {
                text: body,
                bytes,
                truncated,
            }) => column![
                scrollable(text(body).size(13).font(Font::MONOSPACE)).height(Fill),
                text(if *truncated {
                    format!("Preview limited to 96 KiB · {bytes} bytes")
                } else {
                    format!("{bytes} bytes")
                })
                .size(11)
                .color(p.muted)
            ]
            .spacing(8)
            .height(Fill)
            .into(),
            Some(Preview::Html { bytes, .. }) => {
                // The native webview is laid over this pane (webview.rs); the
                // pane carries the palette background so nothing flashes white.
                let mut body = column![].spacing(8).height(Fill);
                if let Some(link) = offered {
                    body = body.push(row![text("This page wants to open").size(12).color(p.muted),
                        text(short(&link, 80)).size(12), Space::new().width(Fill),
                        button(text("Open in browser").size(12)).padding([4, 10]).on_press(Message::OpenOffered(link.clone()))].spacing(6).align_y(Alignment::Center));
                }
                body.push(container(Space::new().width(Fill).height(Fill)).id(super::webview::PANE).width(Fill).height(Fill)
                    .style(move |_| container::Style { background: Some(p.background.into()), ..Default::default() }))
                    .push(text(format!("HTML · {bytes} bytes · scripts run sandboxed to this file's folder")).size(11).color(p.muted))
                    .into()
            }
            Some(Preview::Binary { bytes }) => container(
                text(format!("Binary file · {bytes} bytes"))
                    .size(13)
                    .color(p.muted),
            )
            .center_x(Fill)
            .center_y(Fill)
            .height(Fill)
            .into(),
            None => container(
                text(if self.preview_loading {
                    "Loading preview…"
                } else {
                    "Artifact unavailable"
                })
                .size(13)
                .color(p.muted),
            )
            .center_x(Fill)
            .center_y(Fill)
            .height(Fill)
            .into(),
        };
        content = content.push(body).push(
            row![
                button("Copy path")
                    .on_press(Message::Copy(artifact.id.clone()))
                    .style(button::text),
                Space::new().width(Fill),
                button("Chat")
                    .on_press_maybe(
                        (!running).then(|| Message::OpenSession(artifact.session_id.clone()))
                    )
                    .style(button::text)
            ]
            .spacing(8),
        );
        container(content)
            .padding(20)
            .max_width(960)
            .width(Fill)
            .height(Fill)
            .style(move |_| container::Style {
                background: Some(p.background.into()),
                text_color: Some(p.foreground),
                border: Border {
                    color: p.border,
                    width: 1.0,
                    radius: 10.0.into(),
                },
                ..Default::default()
            })
            .into()
    }
}

fn current_page(page: usize, total: usize, size: usize) -> usize {
    page.max(1).min(total.div_ceil(size).max(1))
}
fn pagination(
    total: usize,
    page: usize,
    size: usize,
    images: bool,
    dark: bool,
) -> Element<'static, Message> {
    let p = super::ui::palette(dark);
    let count = total.div_ceil(size).max(1);
    let action = move |page| {
        if images {
            Message::ImagePage(page)
        } else {
            Message::FilePage(page)
        }
    };
    let mut controls = row![
        Space::new().width(Fill),
        text(format!(
            "{}-{} of {} {}",
            (page - 1) * size + 1,
            (page * size).min(total),
            total,
            if images { "images" } else { "items" }
        ))
        .size(10)
        .color(p.muted)
    ]
    .spacing(8)
    .align_y(Alignment::Center);
    if count > 1 {
        controls = controls.push(
            button("Prev")
                .on_press_maybe((page > 1).then(|| action(page - 1)))
                .padding([3, 6])
                .style(button::text),
        );
        for number in pagination_items(page, count) {
            controls = controls.push(match number {
                Some(number) => button(text(number.to_string()).size(11))
                    .on_press(action(number))
                    .padding([3, 6])
                    .style(button::text),
                None => button(text("…").size(11))
                    .padding([3, 6])
                    .style(button::text),
            });
        }
        controls = controls.push(
            button("Next")
                .on_press_maybe((page < count).then(|| action(page + 1)))
                .padding([3, 6])
                .style(button::text),
        );
    }
    container(controls).height(28).into()
}
fn pagination_items(page: usize, count: usize) -> Vec<Option<usize>> {
    if count <= 7 {
        return (1..=count).map(Some).collect();
    }
    let mut result = vec![Some(1)];
    let start = page.saturating_sub(1).max(2);
    let end = (page + 1).min(count - 1);
    if start > 2 {
        result.push(None);
    }
    result.extend((start..=end).map(Some));
    if end < count - 1 {
        result.push(None);
    }
    result.push(Some(count));
    result
}
fn short(value: &str, limit: usize) -> String {
    let clean = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if clean.chars().count() > limit {
        format!(
            "{}…",
            clean
                .chars()
                .take(limit.saturating_sub(1))
                .collect::<String>()
        )
    } else {
        clean
    }
}
fn semibold() -> Font {
    Font {
        weight: iced::font::Weight::Semibold,
        ..crate::appearance::font()
    }
}
fn time_label(timestamp: f64) -> String {
    Local
        .timestamp_millis_opt(timestamp as i64)
        .single()
        .map(|date| date.format("%b %-d, %H:%M").to_string())
        .unwrap_or_default()
}

fn preview(artifact: &Artifact) -> Result<Preview, String> {
    let mut bytes = vec![];
    let size;
    if artifact.value.starts_with("data:image/") {
        if artifact.value.len() > 24 * 1024 * 1024 {
            return Err("Image exceeds the 16 MiB preview limit".into());
        }
        let (header, payload) = artifact.value.split_once(',').ok_or("Invalid image data")?;
        if !header.ends_with(";base64") {
            return Err("Unsupported inline image encoding".into());
        }
        bytes = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .map_err(|_| "Invalid image data")?;
        size = bytes.len() as u64;
    } else {
        let path = model::local_path(artifact)?;
        let metadata = std::fs::metadata(&path).map_err(|error| error.to_string())?;
        if !metadata.is_file() {
            return Err("Only regular files can be previewed".into());
        }
        size = metadata.len();
        if path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("html") || e.eq_ignore_ascii_case("htm")) {
            let path = path.canonicalize().map_err(|error| error.to_string())?;
            return Ok(Preview::Html { path, bytes: size });
        }
        if artifact.kind == Kind::Image && size > IMAGE_PREVIEW_BYTES {
            return Err("Image exceeds the 16 MiB preview limit".into());
        }
        let file = File::open(path).map_err(|error| error.to_string())?;
        file.take(if artifact.kind == Kind::Image {
            IMAGE_PREVIEW_BYTES + 1
        } else {
            TEXT_PREVIEW_BYTES as u64 + 1
        })
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    }
    if artifact.kind == Kind::Image {
        if bytes.len() as u64 > IMAGE_PREVIEW_BYTES {
            return Err("Image exceeds the 16 MiB preview limit".into());
        }
        if let Ok(mut reader) = ::image::ImageReader::new(Cursor::new(&bytes)).with_guessed_format()
        {
            let mut limits = ::image::Limits::default();
            limits.max_image_width = Some(8192);
            limits.max_image_height = Some(8192);
            limits.max_alloc = Some(64 * 1024 * 1024);
            reader.limits(limits);
            if let Ok(image) = reader.decode() {
                let image = image.to_rgba8();
                let (width, height) = image.dimensions();
                return Ok(Preview::Image {
                    handle: image::Handle::from_rgba(width, height, image.into_raw()),
                    width,
                    height,
                });
            }
        }
        // SVG/unsupported image codecs remain inspectable text/metadata. No
        // HTML, SVG script, external helper, or attachment executable is run.
    }
    let truncated = bytes.len() > TEXT_PREVIEW_BYTES;
    bytes.truncate(TEXT_PREVIEW_BYTES);
    let text = if bytes.contains(&0) {
        None
    } else {
        match std::str::from_utf8(&bytes) {
            Ok(text) => Some(text.to_owned()),
            Err(error) if truncated && error.error_len().is_none() => {
                Some(String::from_utf8_lossy(&bytes[..error.valid_up_to()]).into_owned())
            }
            _ => None,
        }
    };
    Ok(match text {
        Some(text) => Preview::Text {
            text,
            bytes: size,
            truncated,
        },
        None => Preview::Binary { bytes: size },
    })
}

#[cfg(test)]
mod tests;
