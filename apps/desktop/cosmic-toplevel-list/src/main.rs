//! cosmic-toplevel-list — a one-shot COSMIC (cosmic-comp) toplevel enumerator.
//!
//! Hermes Desktop's HUD needs to know which windows are open so it can float
//! *over* them and (on COSMIC) report the window-under awareness. COSMIC does
//! not expose X11-style `_NET_CLIENT_LIST` and runs every app as a native
//! Wayland client, so the only way to enumerate is the Wayland protocol
//! `ext_foreign_toplevel_list_v1` (plus COSMIC's `zcosmic_toplevel_info_v1`).
//!
//! cosmic-comp 1.0 serves `title`, `app_id` and `identifier` over the base
//! `ext_foreign_toplevel_list_v1`. Its `zcosmic_toplevel_info_v1` extension
//! (geometry / state) is advertised but does not emit events in 1.0, so this
//! helper reports geometry as `null` and lets the caller fall back (e.g. run
//! Hermes under XWayland, where `xprop` yields full geometry).
//!
//! Output: a JSON array of `{"title":..,"app_id":..,"identifier":..}`.
//! `--active-only` prints only the currently-focused window (best-effort).

use serde::Serialize;
use wayland_client::{
    Connection, Dispatch, QueueHandle, event_created_child,
    protocol::wl_registry,
};
use wayland_protocols::ext::foreign_toplevel_list::v1::client::{
    ext_foreign_toplevel_handle_v1, ext_foreign_toplevel_list_v1,
};

#[derive(Default)]
struct AppData {
    list: Option<ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1>,
    toplevels: Vec<Toplevel>,
    done: bool,
}

#[derive(Serialize, Clone, Default)]
struct Toplevel {
    title: Option<String>,
    app_id: Option<String>,
    identifier: Option<String>,
    #[serde(skip)]
    handle: Option<ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1>,
}

impl Dispatch<wl_registry::WlRegistry, ()> for AppData {
    fn event(
        _app_data: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _: &(),
        _: &Connection,
        qh: &QueueHandle<AppData>,
    ) {
        if let wl_registry::Event::Global { name, interface, version } = event {
            if interface == "ext_foreign_toplevel_list_v1" {
                _app_data.list = Some(
                    registry.bind::<ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1, _, _>(
                        name, version, qh, (),
                    ),
                );
            }
        }
    }
}

impl Dispatch<ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1, ()> for AppData {
    fn event(
        app_data: &mut Self,
        _list: &ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1,
        event: ext_foreign_toplevel_list_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<AppData>,
    ) {
        match event {
            ext_foreign_toplevel_list_v1::Event::Toplevel { toplevel } => {
                app_data.toplevels.push(Toplevel {
                    title: None,
                    app_id: None,
                    identifier: None,
                    handle: Some(toplevel),
                });
            }
            ext_foreign_toplevel_list_v1::Event::Finished => {
                app_data.done = true;
            }
            _ => {}
        }
    }

    event_created_child!(
        AppData,
        ext_foreign_toplevel_list_v1::ExtForeignToplevelListV1,
        [
            ext_foreign_toplevel_list_v1::EVT_TOPLEVEL_OPCODE => (ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1, ()),
        ]
    );
}

impl Dispatch<ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1, ()> for AppData {
    fn event(
        app_data: &mut Self,
        toplevel: &ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1,
        event: ext_foreign_toplevel_handle_v1::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<AppData>,
    ) {
        let info = match app_data.toplevels.iter_mut().find(|t| t.handle.as_ref() == Some(toplevel)) {
            Some(i) => i,
            None => return,
        };
        match event {
            ext_foreign_toplevel_handle_v1::Event::Title { title } => info.title = Some(title),
            ext_foreign_toplevel_handle_v1::Event::AppId { app_id } => info.app_id = Some(app_id),
            ext_foreign_toplevel_handle_v1::Event::Identifier { identifier } => {
                info.identifier = Some(identifier)
            }
            _ => {}
        }
    }

    event_created_child!(
        AppData,
        ext_foreign_toplevel_handle_v1::ExtForeignToplevelHandleV1,
        []
    );
}

fn main() {
    let active_only = std::env::args().any(|a| a == "--active-only");

    let conn = match Connection::connect_to_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("cosmic-toplevel-list: cannot connect to Wayland: {e}");
            std::process::exit(2);
        }
    };
    let mut event_queue = conn.new_event_queue();
    let qh = event_queue.handle();
    let _registry = conn.display().get_registry(&qh, ());

    let mut app_data = AppData::default();
    let _ = event_queue.roundtrip(&mut app_data);
    let _ = event_queue.roundtrip(&mut app_data);
    let start = std::time::Instant::now();
    while !app_data.done && start.elapsed() < std::time::Duration::from_millis(1500) {
        if event_queue.dispatch_pending(&mut app_data).unwrap_or(0) == 0 {
            let _ = conn.flush();
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }

    let out: Vec<serde_json::Value> = app_data
        .toplevels
        .into_iter()
        .map(|t| {
            serde_json::json!({
                "title": t.title,
                "app_id": t.app_id,
                "identifier": t.identifier,
                "geometry": null, // COSMIC 1.0 does not serve geometry over Wayland
            })
        })
        .collect();

    if active_only {
        // Best-effort: the first window reported by the compositor is typically
        // the focused one; callers should prefer XWayland for true geometry.
        if let Some(first) = out.into_iter().next() {
            println!("{}", serde_json::to_string_pretty(&first).unwrap());
        }
    } else {
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    }
}
