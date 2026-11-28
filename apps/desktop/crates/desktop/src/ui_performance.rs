//! Opt-in comparison of the native view/layout/draw path for composer edits.
use super::*;
use iced_core::renderer::Headless;
use std::time::{Duration, Instant};

#[tokio::test]
#[ignore = "uses a fresh isolated profile and rasterizes several frames"]
async fn composer_frame_timings() {
    let dir=std::env::var_os("HERMES_RUST_DATA_DIR").map(PathBuf::from).expect("Set HERMES_RUST_DATA_DIR to an empty temporary directory");
    assert!(dir.is_absolute() && dir.is_dir() && dir.read_dir().unwrap().next().is_none());
    assert!(std::env::var_os("HERMES_HOME").is_some(),"Set HERMES_HOME to an isolated empty path");
    let (mut app,_)=Hermes::new();
    let mut renderer=<iced::Renderer as Headless>::new(iced::Font::DEFAULT,16.into(),None).await.unwrap();
    let mut cache=iced_runtime::user_interface::Cache::new();
    let mut layouts=Vec::new(); let mut frames=Vec::new();
    let theme=ui::theme_for(false);
    for i in 0..11 {
        let started=Instant::now();
        let _=app.update(Message::Edit(text_editor::Action::Edit(text_editor::Edit::Insert('x'))));
        let mut tree=iced_runtime::UserInterface::build(app.view(),iced::Size::new(1220.0,800.0),cache,&mut renderer);
        tree.draw(&mut renderer,&theme,&iced_core::renderer::Style::default(),iced::mouse::Cursor::Unavailable);
        let layout=started.elapsed();
        let pixels=Headless::screenshot(&mut renderer,iced::Size::new(1220,800),1.0,iced::Color::WHITE);
        assert_eq!(pixels.len(),1220*800*4);
        let elapsed=started.elapsed();
        cache=tree.into_cache();
        if i>0 { layouts.push(layout); frames.push(elapsed); }
    }
    fn report(name:&str, mut samples:Vec<Duration>) {
        samples.sort();
        eprintln!("{name}: median {:.2} ms, maximum {:.2} ms (10 warmed edits)",samples[5].as_secs_f64()*1000.0,samples[9].as_secs_f64()*1000.0);
    }
    report("edit + layout + draw preparation",layouts);
    report("edit through full 1220x800 raster readback",frames);
}
