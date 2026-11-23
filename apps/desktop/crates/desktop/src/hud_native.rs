//! The HUD window's platform layer (Electron: `transparent` + `vibrancy`,
//! `setIgnoreMouseEvents(…, {forward:true})` and the band's opacity fade).
//! Frost is a system material composited below the transparent wgpu surface:
//! NSVisualEffectView on macOS, the DWM transient (acrylic) backdrop on
//! Windows 11. The fade is the window's own alpha, so the frost fades with the
//! band instead of outliving it. While faded the window passes clicks through;
//! the cursor is then read globally to tell when it comes back over the band.
//! Linux compositors offer no portable material: the band stays opaque there.
use iced::window;
use raw_window_handle::RawWindowHandle;

fn raw(window: &dyn window::Window) -> Option<RawWindowHandle> { window.window_handle().ok().map(|h| h.as_raw()) }

#[cfg(target_os = "macos")]
mod mac {
    use objc2::{msg_send, runtime::{AnyClass, AnyObject}};
    use std::ffi::c_void;
    #[repr(C)] #[derive(Clone, Copy)] struct Point { x: f64, y: f64 }
    #[repr(C)] #[derive(Clone, Copy)] struct Size { w: f64, h: f64 }
    #[repr(C)] #[derive(Clone, Copy)] struct Rect { origin: Point, size: Size }
    unsafe impl objc2::Encode for Point { const ENCODING: objc2::Encoding = objc2::Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]); }
    unsafe impl objc2::Encode for Size { const ENCODING: objc2::Encoding = objc2::Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]); }
    unsafe impl objc2::Encode for Rect { const ENCODING: objc2::Encoding = objc2::Encoding::Struct("CGRect", &[Point::ENCODING, Size::ENCODING]); }

    unsafe fn ns_window(view: *mut c_void) -> Option<&'static AnyObject> {
        let view = unsafe { (view as *mut AnyObject).as_ref()? };
        let window: *mut AnyObject = unsafe { msg_send![view, window] };
        unsafe { window.as_ref() }
    }
    /// Put a HUD-material NSVisualEffectView under the content view (once).
    pub fn frost(view: *mut c_void) {
        unsafe {
            let Some(window) = ns_window(view) else { return };
            let content: *mut AnyObject = msg_send![window, contentView];
            let Some(content) = content.as_ref() else { return };
            let superview: *mut AnyObject = msg_send![content, superview];
            let Some(superview) = superview.as_ref() else { return };
            let Some(class) = AnyClass::get("NSVisualEffectView") else { return };
            let subviews: *mut AnyObject = msg_send![superview, subviews];
            let count: usize = msg_send![subviews, count];
            for i in 0..count {
                let child: *mut AnyObject = msg_send![subviews, objectAtIndex: i];
                let is: bool = msg_send![child, isKindOfClass: class];
                if is { return }
            }
            let bounds: Rect = msg_send![content, frame];
            let fx: *mut AnyObject = msg_send![class, alloc];
            let fx: *mut AnyObject = msg_send![fx, initWithFrame: bounds];
            let _: () = msg_send![fx, setMaterial: 13isize]; // NSVisualEffectMaterialHUDWindow
            let _: () = msg_send![fx, setBlendingMode: 0isize]; // behindWindow
            let _: () = msg_send![fx, setState: 1isize]; // active, also while the window is blurred
            let _: () = msg_send![fx, setAutoresizingMask: 18usize]; // width | height sizable
            let _: () = msg_send![fx, setWantsLayer: true];
            let layer: *mut AnyObject = msg_send![fx, layer];
            if let Some(layer) = layer.as_ref() {
                let _: () = msg_send![layer, setCornerRadius: 12.0f64];
                let _: () = msg_send![layer, setMasksToBounds: true];
            }
            let _: () = msg_send![superview, addSubview: fx, positioned: -1isize, relativeTo: content]; // NSWindowBelow
            let _: () = msg_send![window, setOpaque: false];
            let _: () = msg_send![window, setHasShadow: false];
        }
    }
    pub fn alpha(view: *mut c_void, value: f64) {
        unsafe { if let Some(window) = ns_window(view) { let _: () = msg_send![window, setAlphaValue: value]; } }
    }
    pub fn cursor_inside(view: *mut c_void) -> Option<bool> {
        unsafe {
            let window = ns_window(view)?;
            let key: bool = msg_send![window, isKeyWindow];
            if key { return Some(true) }
            let frame: Rect = msg_send![window, frame];
            let event = AnyClass::get("NSEvent")?;
            let at: Point = msg_send![event, mouseLocation];
            Some(at.x >= frame.origin.x && at.x <= frame.origin.x + frame.size.w && at.y >= frame.origin.y && at.y <= frame.origin.y + frame.size.h)
        }
    }
}

#[cfg(target_os = "windows")]
mod win {
    use windows_sys::Win32::{Foundation::{HWND, POINT, RECT}, Graphics::Dwm::DwmSetWindowAttribute, UI::WindowsAndMessaging::*};
    pub fn frost(hwnd: isize) {
        let hwnd = hwnd as HWND;
        let backdrop: i32 = 3; // DWMSBT_TRANSIENTWINDOW (acrylic)
        let corner: i32 = 2; // DWMWCP_ROUND
        unsafe {
            DwmSetWindowAttribute(hwnd, 38, &backdrop as *const _ as _, 4); // DWMWA_SYSTEMBACKDROP_TYPE
            DwmSetWindowAttribute(hwnd, 33, &corner as *const _ as _, 4); // DWMWA_WINDOW_CORNER_PREFERENCE
        }
    }
    pub fn alpha(hwnd: isize, value: f64) {
        let hwnd = hwnd as HWND;
        unsafe {
            let style = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
            if style & WS_EX_LAYERED as isize == 0 { SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED as isize); }
            SetLayeredWindowAttributes(hwnd, 0, (value.clamp(0.0, 1.0) * 255.0) as u8, LWA_ALPHA);
        }
    }
    pub fn cursor_inside(hwnd: isize) -> Option<bool> {
        if unsafe { GetForegroundWindow() } == hwnd as HWND { return Some(true) }
        let mut at = POINT { x: 0, y: 0 };
        let mut rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        unsafe { (GetCursorPos(&mut at) != 0 && GetWindowRect(hwnd as HWND, &mut rect) != 0).then(|| at.x >= rect.left && at.x < rect.right && at.y >= rect.top && at.y < rect.bottom) }
    }
}

/// Engage the platform material behind the band (once per window).
pub(super) fn frost(window: &dyn window::Window) {
    match raw(window) {
        #[cfg(target_os = "macos")] Some(RawWindowHandle::AppKit(h)) => mac::frost(h.ns_view.as_ptr()),
        #[cfg(target_os = "windows")] Some(RawWindowHandle::Win32(h)) => win::frost(h.hwnd.get()),
        _ => {}
    }
}
/// Set the whole window's opacity (band and frost together).
pub(super) fn alpha(window: &dyn window::Window, value: f32) {
    match raw(window) {
        #[cfg(target_os = "macos")] Some(RawWindowHandle::AppKit(h)) => mac::alpha(h.ns_view.as_ptr(), value as f64),
        #[cfg(target_os = "windows")] Some(RawWindowHandle::Win32(h)) => win::alpha(h.hwnd.get(), value as f64),
        _ => { let _ = value; }
    }
}
/// Whether the user is at the band: it holds keyboard focus (the caret is in its composer),
/// or the cursor is over it, read globally so it works while clicks pass through.
pub(super) fn cursor_inside(window: &dyn window::Window) -> Option<bool> {
    match raw(window) {
        #[cfg(target_os = "macos")] Some(RawWindowHandle::AppKit(h)) => mac::cursor_inside(h.ns_view.as_ptr()),
        #[cfg(target_os = "windows")] Some(RawWindowHandle::Win32(h)) => win::cursor_inside(h.hwnd.get()),
        _ => None,
    }
}
/// Whether this platform can fade and pass clicks through (Linux keeps a solid band).
pub(super) const FADES: bool = cfg!(any(target_os = "macos", target_os = "windows"));

/// ui-test: post a real click at (x, y) (top-left window points) and then `text` as key presses
/// into the window through NSApplication's event queue, exactly as the window server would.
#[cfg(all(feature = "ui-test", target_os = "macos"))]
pub(super) fn synthesize(window: &dyn window::Window, x: f64, y: f64, text: &str) {
    use objc2::{msg_send, runtime::{AnyClass, AnyObject}};
    #[repr(C)] #[derive(Clone, Copy)] struct P { x: f64, y: f64 }
    unsafe impl objc2::Encode for P { const ENCODING: objc2::Encoding = objc2::Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]); }
    use objc2::Encode;
    let Some(RawWindowHandle::AppKit(h)) = raw(window) else { return };
    unsafe {
        let view = h.ns_view.as_ptr() as *mut AnyObject;
        let win: *mut AnyObject = msg_send![view, window];
        let Some(win) = win.as_ref() else { return };
        let number: isize = msg_send![win, windowNumber];
        let content: *mut AnyObject = msg_send![win, contentView];
        #[repr(C)] #[derive(Clone, Copy)] struct S { w: f64, h: f64 }
        #[repr(C)] #[derive(Clone, Copy)] struct R { o: P, s: S }
        unsafe impl objc2::Encode for S { const ENCODING: objc2::Encoding = objc2::Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]); }
        unsafe impl objc2::Encode for R { const ENCODING: objc2::Encoding = objc2::Encoding::Struct("CGRect", &[P::ENCODING, S::ENCODING]); }
        let frame: R = msg_send![content, frame];
        let at = P { x, y: frame.s.h - y };
        let event = AnyClass::get("NSEvent").unwrap();
        let app_class = AnyClass::get("NSApplication").unwrap();
        let app: *mut AnyObject = msg_send![app_class, sharedApplication];
        let null: *mut AnyObject = std::ptr::null_mut();
        for kind in [1usize, 2usize] { // leftMouseDown, leftMouseUp
            let e: *mut AnyObject = msg_send![event, mouseEventWithType: kind, location: at, modifierFlags: 0usize, timestamp: 0.0f64, windowNumber: number, context: null, eventNumber: 0isize, clickCount: 1isize, pressure: 1.0f32];
            let _: () = msg_send![app, postEvent: e, atStart: false];
        }
        let ns_string = AnyClass::get("NSString").unwrap();
        for ch in text.chars() {
            let s = std::ffi::CString::new(ch.to_string()).unwrap();
            let chars: *mut AnyObject = msg_send![ns_string, stringWithUTF8String: s.as_ptr()];
            for kind in [10usize, 11usize] { // keyDown, keyUp
                let e: *mut AnyObject = msg_send![event, keyEventWithType: kind, location: at, modifierFlags: 0usize, timestamp: 0.0f64, windowNumber: number, context: null, characters: chars, charactersIgnoringModifiers: chars, isARepeat: false, keyCode: 4u16];
                let _: () = msg_send![app, postEvent: e, atStart: false];
            }
        }
    }
}
