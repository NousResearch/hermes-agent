//! Windows Common Item Dialog / Linux XDG Desktop Portal. Callers use a
//! worker thread so a modal dialog never blocks rendering or streaming.
use std::path::PathBuf;

pub(super) fn choose(title:&str,directory:bool)->Result<Option<PathBuf>,String>{
    let dialog=rfd::FileDialog::new().set_title(title);
    Ok(if directory {dialog.pick_folder()}else{dialog.pick_file()})
}
pub(super) fn save(title:&str,filename:&str)->Result<Option<PathBuf>,String>{
    Ok(rfd::FileDialog::new().set_title(title).set_file_name(filename).save_file())
}
