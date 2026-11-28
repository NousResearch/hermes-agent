//! Star map (app/starmap): a top-down map of what Hermes has learned for this
//! profile — skills as stars clustered by category around the core, linked by
//! their related skills, with freeform memories listed beside it. Hover to
//! read a star, click to inspect it; share codes export or load a map.
use super::{Hermes, Message, ui};
use crate::appearance::text;
use hermes_engine::starmap::{Graph, Node};
use iced::{Alignment, Color, Element, Fill, Point, Rectangle, Renderer, Task, Theme, Vector, mouse,
    widget::{Space, button, canvas, column, container, row, scrollable, text_input}};

#[derive(Debug, Clone)]
pub(super) enum Action { Refresh, Select(Option<String>), CopyShare, ShareInput(String), LoadShare, Mine, OpenCapabilities }

#[derive(Default)]
pub(super) struct State { graph: Graph, imported: bool, selected: Option<String>, share: String, error: Option<String> }

#[cfg(test)]
impl Hermes { pub(super) fn starmap_first(&self) -> Option<String> { self.starmap.graph.nodes.first().map(|n| n.id.clone()) } }

pub(super) fn update(app: &mut Hermes, action: Action) -> Task<Message> {
    use base64::Engine;
    let root = app.library.as_ref().map(|l| l.root().to_owned());
    let state = &mut app.starmap;
    match action {
        Action::Refresh | Action::Mine => { if let Some(root) = &root { state.graph = hermes_engine::starmap::graph(root); } state.imported = false; state.error = None; }
        Action::Select(id) => state.selected = id,
        Action::CopyShare => { let code = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&state.graph).unwrap_or_default()); return iced::clipboard::write(format!("hermes-starmap:{code}")); }
        Action::ShareInput(v) => state.share = v,
        Action::LoadShare => {
            let code = state.share.trim().trim_start_matches("hermes-starmap:");
            match base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(code).ok().and_then(|b| serde_json::from_slice::<Graph>(&b).ok()) {
                Some(graph) if graph.nodes.len() <= 20_000 => { state.graph = graph; state.imported = true; state.selected = None; state.share.clear(); state.error = None; }
                _ => state.error = Some("That is not a star map share code.".into()),
            }
        }
        Action::OpenCapabilities => return app.update(Message::Navigate(super::Screen::Tools)),
    }
    Task::none()
}

fn hue(name: &str) -> Color {
    let h = name.bytes().fold(2166136261u32, |a, b| (a ^ b as u32).wrapping_mul(16777619)) % 360;
    let (h, s, l) = (h as f32 / 60.0, 0.62, 0.62);
    let c = (1.0 - (2.0 * l - 1.0f32).abs()) * s; let x = c * (1.0 - (h % 2.0 - 1.0).abs()); let m = l - c / 2.0;
    let (r, g, b) = match h as u32 { 0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x), 3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x) };
    Color::from_rgb(r + m, g + m, b + m)
}

/// Star positions inside `bounds`: clusters on a ring, stars on a golden-angle spiral per cluster.
fn layout(graph: &Graph, bounds: Rectangle) -> Vec<(Point, usize)> {
    let center = Point::new(bounds.width / 2.0, bounds.height / 2.0);
    let ring = bounds.width.min(bounds.height) * 0.33;
    let clusters: Vec<&str> = graph.clusters.iter().map(|c| c.0.as_str()).collect();
    let mut counts = vec![0usize; clusters.len()];
    graph.nodes.iter().map(|node| {
        let ci = clusters.iter().position(|c| *c == node.category).unwrap_or(0);
        let angle = ci as f32 / clusters.len().max(1) as f32 * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
        let size = graph.clusters.get(ci).map_or(1, |c| c.1) as f32;
        let k = counts[ci] as f32; counts[ci] += 1;
        let spread = (ring * 0.32).min(14.0 + size.sqrt() * 9.0);
        let r = spread * (k / size.max(1.0)).sqrt();
        let a = k * 2.399_963;
        (Point::new(center.x + ring * angle.cos() + r * a.cos(), center.y + ring * angle.sin() + r * a.sin()), ci)
    }).collect()
}

struct Map<'a> { graph: &'a Graph, selected: Option<&'a str>, dark: bool }
#[derive(Default)]
struct Hover(Option<usize>);

impl canvas::Program<Message> for Map<'_> {
    type State = Hover;
    fn update(&self, state: &mut Hover, event: &canvas::Event, bounds: Rectangle, cursor: mouse::Cursor) -> Option<canvas::Action<Message>> {
        let positions = layout(self.graph, bounds);
        let nearest = cursor.position_in(bounds).and_then(|p| positions.iter().enumerate()
            .map(|(i, (q, _))| (i, (q.x - p.x).hypot(q.y - p.y))).filter(|(_, d)| *d < 9.0).min_by(|a, b| a.1.total_cmp(&b.1)).map(|(i, _)| i));
        match event {
            canvas::Event::Mouse(mouse::Event::CursorMoved { .. }) if nearest != state.0 => { state.0 = nearest; Some(canvas::Action::request_redraw()) }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) if cursor.is_over(bounds) =>
                Some(canvas::Action::publish(Message::Starmap(Action::Select(nearest.map(|i| self.graph.nodes[i].id.clone())))).and_capture()),
            _ => None,
        }
    }
    fn draw(&self, state: &Hover, renderer: &Renderer, _theme: &Theme, bounds: Rectangle, _cursor: mouse::Cursor) -> Vec<canvas::Geometry> {
        let p = ui::palette(self.dark);
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let positions = layout(self.graph, bounds);
        let center = Point::new(bounds.width / 2.0, bounds.height / 2.0);
        let ring = bounds.width.min(bounds.height) * 0.33;
        // Time-axis rings and the core.
        for f in [0.5f32, 1.0, 1.5] {
            frame.stroke(&canvas::Path::circle(center, ring * f), canvas::Stroke::default().with_color(Color { a: 0.07, ..p.foreground }).with_width(1.0));
        }
        frame.fill(&canvas::Path::circle(center, 7.0), p.accent);
        frame.fill_text(canvas::Text { content: "Hermes".into(), position: center + Vector::new(0.0, 12.0), color: p.muted, size: 11.0.into(),
            align_x: iced::advanced::text::Alignment::Center, ..Default::default() });
        let index = |id: &str| self.graph.nodes.iter().position(|n| n.id == id);
        let focus = state.0.map(|i| self.graph.nodes[i].id.as_str()).or(self.selected);
        for (a, b) in &self.graph.edges {
            let (Some(i), Some(j)) = (index(a), index(b)) else { continue };
            let lit = focus.is_some_and(|f| f == a || f == b);
            frame.stroke(&canvas::Path::line(positions[i].0, positions[j].0), canvas::Stroke::default().with_width(if lit { 1.4 } else { 0.8 })
                .with_color(Color { a: if lit { 0.7 } else { 0.16 }, ..p.accent }));
        }
        for (ci, (name, count)) in self.graph.clusters.iter().enumerate() {
            let angle = ci as f32 / self.graph.clusters.len().max(1) as f32 * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
            let at = center + Vector::new((ring * 1.38) * angle.cos(), (ring * 1.38) * angle.sin());
            frame.fill_text(canvas::Text { content: format!("{} · {count}", name.to_uppercase()), position: at, color: hue(name), size: 10.0.into(),
                align_x: iced::advanced::text::Alignment::Center, align_y: iced::alignment::Vertical::Center, ..Default::default() });
        }
        for (i, (point, ci)) in positions.iter().enumerate() {
            let node = &self.graph.nodes[i];
            let color = hue(&self.graph.clusters.get(*ci).map_or(String::new(), |c| c.0.clone()));
            let active = focus == Some(node.id.as_str());
            if active { frame.fill(&canvas::Path::circle(*point, 8.0), Color { a: 0.25, ..color }); }
            frame.fill(&canvas::Path::circle(*point, if active { 4.5 } else { 3.0 }), color);
        }
        if let Some(i) = state.0 {
            let (point, _) = positions[i];
            frame.fill_text(canvas::Text { content: self.graph.nodes[i].label.clone(), position: point + Vector::new(9.0, -6.0), color: p.foreground, size: 12.0.into(), ..Default::default() });
        }
        vec![frame.into_geometry()]
    }
    fn mouse_interaction(&self, state: &Hover, _bounds: Rectangle, _cursor: mouse::Cursor) -> mouse::Interaction {
        if state.0.is_some() { mouse::Interaction::Pointer } else { mouse::Interaction::default() }
    }
}

pub(super) fn view(app: &Hermes) -> Element<'_, Message> {
    let state = &app.starmap; let p = ui::palette(app.dark);
    let act = Message::Starmap;
    let g = &state.graph;
    let mut header = row![text(format!("{} skills · {} categories · {} related links · {} memories", g.nodes.len(), g.clusters.len(), g.edges.len(), g.memory.len())).size(12).color(p.muted).width(Fill)].spacing(8).align_y(Alignment::Center);
    if state.imported { header = header.push(button(text("Back to my map").size(11)).style(ui::quiet_button).on_press(act(Action::Mine))); }
    header = header.push(button(text("Copy share code").size(11)).style(ui::quiet_button).on_press(act(Action::CopyShare)))
        .push(text_input("Paste a share code…", &state.share).on_input(move |v| act(Action::ShareInput(v))).on_submit(act(Action::LoadShare)).size(12).padding([4, 8]).width(200));
    let map: Element<'_, Message> = if g.nodes.is_empty() {
        container(column![ui::icon("sparkle", 22, p.muted), text("Nothing learned yet").size(14), text("Enable or install skills and Hermes' map fills in.").size(12).color(p.muted)].spacing(6).align_x(Alignment::Center)).center(Fill).into()
    } else { canvas(Map { graph: g, selected: state.selected.as_deref(), dark: app.dark }).width(Fill).height(Fill).into() };
    let mut side = column![].spacing(10).width(300);
    if let Some(node) = state.selected.as_ref().and_then(|id| g.nodes.iter().find(|n| &n.id == id)) {
        side = side.push(detail(node, g, app.dark));
    }
    side = side.push(text("MEMORY").size(11).font(ui::medium()).color(p.muted));
    if g.memory.is_empty() { side = side.push(text("No saved memories yet.").size(12).color(p.muted)); }
    for card in &g.memory {
        side = side.push(container(column![text(if card.source == "profile" { "About you" } else { "Memory" }).size(10).color(p.muted), text(card.text.clone()).size(12)].spacing(3))
            .padding(10).width(Fill).style(move |_| ui::panel(p.card, 6.0, Some(p.border))));
    }
    let mut body = column![header].spacing(10).padding([8, 24]);
    if let Some(e) = &state.error { body = body.push(text(e.clone()).size(12).color(Color::from_rgb8(0xd1, 0x43, 0x43))); }
    body.push(row![container(map).width(Fill).height(Fill).style(move |_| ui::panel(Color { a: 0.02, ..p.foreground }, 8.0, Some(p.border))), scrollable(side).height(Fill)].spacing(16).height(Fill)).height(Fill).into()
}

fn detail<'a>(node: &'a Node, graph: &'a Graph, dark: bool) -> Element<'a, Message> {
    let p = ui::palette(dark);
    let links: Vec<&str> = graph.edges.iter().filter_map(|(a, b)| if a == &node.id { Some(b) } else if b == &node.id { Some(a) } else { None })
        .filter_map(|id| graph.nodes.iter().find(|n| &n.id == id).map(|n| n.label.as_str())).collect();
    let mut col = column![
        row![container(Space::new().width(8).height(8)).style(move |_| ui::panel(hue(&node.category), 4.0, None)), text(node.label.clone()).size(15).font(ui::medium())].spacing(8).align_y(Alignment::Center),
        text(format!("Skill · {}", node.category)).size(11).color(p.muted),
    ].spacing(6);
    if !node.description.is_empty() { col = col.push(text(node.description.clone()).size(12)); }
    if !links.is_empty() { col = col.push(text(format!("Related: {}", links.join(", "))).size(11).color(p.muted)); }
    col = col.push(row![button(text("Open in Capabilities").size(11)).style(ui::quiet_button).on_press(Message::Starmap(Action::OpenCapabilities)), Space::new().width(Fill),
        button(text("Close").size(11)).style(ui::quiet_button).on_press(Message::Starmap(Action::Select(None)))]);
    container(col).padding(12).width(Fill).style(move |_| ui::panel(p.card, 8.0, Some(Color { a: 0.4, ..p.accent }))).into()
}
