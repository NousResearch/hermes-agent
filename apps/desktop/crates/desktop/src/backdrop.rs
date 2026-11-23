//! Rust rendering of the public chat Backdrop. The original JPEG is decoded
//! away from the event loop once and retained under one image handle.
use iced::widget::image;
use std::{path::Path, sync::OnceLock};
static TEXTURE: OnceLock<Result<image::Handle,String>> = OnceLock::new();

pub(super) fn texture() -> Result<image::Handle,String> {
    TEXTURE.get_or_init(||{
        let asset=include_bytes!("../../../public/ds-assets/filler-bg0.jpg");
        let picture=::image::load_from_memory_with_format(asset,::image::ImageFormat::Jpeg).map_err(|e|e.to_string())?.into_rgba8();
        Ok(image::Handle::from_rgba(picture.width(),picture.height(),picture.into_raw()))
    }).clone()
}

pub(super) fn enabled(directory:&Path) -> bool {
    std::fs::read_to_string(directory.join("chat-backdrop")).ok().map(|s|s.trim()=="on")
        .unwrap_or_else(||std::env::var_os("HERMES_RUST_BACKDROP").as_deref()==Some(std::ffi::OsStr::new("1")))
}
pub(super) fn save(directory:&Path,enabled:bool)->Result<(),String>{
    use std::io::Write;
    let mut temp=tempfile::NamedTempFile::new_in(directory).map_err(|e|e.to_string())?;
    temp.write_all(if enabled {b"on\n"} else {b"off\n"}).map_err(|e|e.to_string())?;
    temp.persist(directory.join("chat-backdrop")).map_err(|e|e.to_string())?;
    Ok(())
}
