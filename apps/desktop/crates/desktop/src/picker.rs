//! Native AppKit open panel shared by project selection and history import.
use objc2_app_kit::{NSModalResponseOK, NSOpenPanel};
use objc2_foundation::{MainThreadMarker, NSString};
use std::{
    ffi::{CStr, OsStr},
    os::unix::ffi::OsStrExt,
    path::PathBuf,
};

pub(super) fn choose(prompt: &'static str, directories: bool) -> Result<Option<PathBuf>, String> {
    fn show(
        mtm: MainThreadMarker,
        prompt: &str,
        directories: bool,
    ) -> Result<Option<PathBuf>, String> {
        // AppKit confines panels to the main thread. The selected NSURL owns the
        // file-system representation until we copy the path into a Rust PathBuf.
        unsafe {
            let panel = NSOpenPanel::openPanel(mtm);
            panel.setMessage(Some(&NSString::from_str(prompt)));
            panel.setCanChooseDirectories(directories);
            panel.setCanChooseFiles(!directories);
            panel.setAllowsMultipleSelection(false);
            if panel.runModal() != NSModalResponseOK {
                return Ok(None);
            }
            let url = panel.URL().ok_or("The file picker returned no location")?;
            let path = CStr::from_ptr(url.fileSystemRepresentation().as_ptr());
            Ok(Some(PathBuf::from(OsStr::from_bytes(path.to_bytes()))))
        }
    }
    if let Some(main) = MainThreadMarker::new() {
        return show(main, prompt, directories);
    }
    dispatch::Queue::main().exec_sync(move || {
        let main =
            MainThreadMarker::new().ok_or("Could not open the file picker on the main thread")?;
        show(main, prompt, directories)
    })
}
