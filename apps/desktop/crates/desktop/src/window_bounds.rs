//! Stock desktop/electron/window-state.ts normal size calculation in macOS
//! logical points. Saved bounds belong to this Rust development instance.
use iced::{Point, Size, window::Position};
use objc2_app_kit::{NSApplication,NSScreen,NSWindowStyleMask};
use objc2_foundation::MainThreadMarker;
use serde::{Deserialize,Serialize};
use std::path::Path;

#[derive(Serialize,Deserialize)]
struct Bounds {x:f64,y:f64,width:f64,height:f64}
impl Bounds{
    fn valid(&self)->bool{[self.x,self.y,self.width,self.height].into_iter().all(f64::is_finite)&&self.width>=400.&&self.height>=620.}
}
fn screens(main:MainThreadMarker)->Vec<Bounds>{
    let screens=NSScreen::screens(main);
    let top=screens.iter().next().map(|s|s.frame().size.height).unwrap_or(0.);
    screens.iter().map(|screen|{
        let r=screen.visibleFrame();
        Bounds{x:r.origin.x,y:top-r.origin.y-r.size.height,width:r.size.width,height:r.size.height}
    }).collect()
}
pub(super) fn initial(root:Option<&Path>)->(Size,Position){
    let areas=MainThreadMarker::new().map(screens).unwrap_or_default();
    let saved=root.and_then(|root|std::fs::File::open(root.join("desktop-window.json")).ok())
        .and_then(|file|serde_json::from_reader::<_,Bounds>(file).ok()).filter(Bounds::valid);
    if let Some(bounds)=saved && let Some(area)=areas.iter().find(|a|
        bounds.x+bounds.width>=a.x+48. && bounds.x<=a.x+a.width-48. && bounds.y>=a.y-1. && bounds.y<a.y+a.height-48.){
        let size=Size::new(bounds.width.min(area.width) as f32,bounds.height.min(area.height) as f32);
        return (size,Position::Specific(Point::new(bounds.x as f32,bounds.y as f32)));
    }
    if let Some(area)=areas.first(){
        let size=Size::new((area.width*0.88).round().clamp(1280.,1760.).min(area.width) as f32,
            (area.height*0.88).round().clamp(820.,1100.).min(area.height) as f32);
        return (size,Position::Centered);
    }
    (Size::new(1280.,820.),Position::Centered)
}
pub(super) fn save(root:&Path)->anyhow::Result<()>{
    let Some(main)=MainThreadMarker::new()else{return Ok(())};
    let Some(window)=(unsafe{NSApplication::sharedApplication(main).mainWindow()})else{return Ok(())};
    if window.isZoomed() || window.styleMask().contains(NSWindowStyleMask::FullScreen){return Ok(());}
    let Some(primary)=NSScreen::screens(main).iter().next().map(|s|s.frame())else{return Ok(())};
    let rect=window.frame();
    let bounds=Bounds{x:rect.origin.x,y:primary.size.height-rect.origin.y-rect.size.height,width:rect.size.width,height:rect.size.height};
    if !bounds.valid(){return Ok(())}
    let mut tmp=tempfile::NamedTempFile::new_in(root)?;
    serde_json::to_writer(&mut tmp,&bounds)?;tmp.as_file().sync_all()?;tmp.persist(root.join("desktop-window.json"))?;
    Ok(())
}
