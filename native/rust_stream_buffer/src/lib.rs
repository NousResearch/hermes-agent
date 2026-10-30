//! Small C ABI stream accumulator used by the Python gateway.
//!
//! The gateway still owns delivery policy and rendering. This library only
//! stores two byte buffers: the active segment and the complete turn ledger.
//! Keeping the append operation native avoids repeatedly reallocating Python
//! strings for every streamed delta.

#![allow(clippy::missing_safety_doc)] // C ABI safety contract: pointers originate from the Python wrapper.

use std::ptr;

#[repr(C)]
pub struct HermesStreamBuffer {
    active: Vec<u8>,
    ledger: Vec<u8>,
}

unsafe fn input_slice<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if data.is_null() {
        return None;
    }
    Some(std::slice::from_raw_parts(data, len))
}

unsafe fn output_copy(data: &[u8], out: *mut u8, capacity: usize) -> usize {
    let count = data.len().min(capacity);
    if count > 0 && !out.is_null() {
        ptr::copy_nonoverlapping(data.as_ptr(), out, count);
    }
    data.len()
}

fn replace_bytes(target: &mut Vec<u8>, bytes: &[u8]) -> bool {
    if target.try_reserve(bytes.len()).is_err() {
        return false;
    }
    target.clear();
    target.extend_from_slice(bytes);
    true
}

#[no_mangle]
pub extern "C" fn hermes_stream_buffer_new() -> *mut HermesStreamBuffer {
    Box::into_raw(Box::new(HermesStreamBuffer {
        active: Vec::new(),
        ledger: Vec::new(),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_free(buffer: *mut HermesStreamBuffer) {
    if !buffer.is_null() {
        drop(Box::from_raw(buffer));
    }
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_append(
    buffer: *mut HermesStreamBuffer,
    data: *const u8,
    len: usize,
) -> i32 {
    let Some(bytes) = input_slice(data, len) else {
        return 0;
    };
    let Some(state) = buffer.as_mut() else {
        return 0;
    };
    if state.active.try_reserve(bytes.len()).is_err()
        || state.ledger.try_reserve(bytes.len()).is_err()
    {
        return 0;
    }
    state.active.extend_from_slice(bytes);
    state.ledger.extend_from_slice(bytes);
    1
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_set_active(
    buffer: *mut HermesStreamBuffer,
    data: *const u8,
    len: usize,
) -> i32 {
    let Some(bytes) = input_slice(data, len) else {
        return 0;
    };
    let Some(state) = buffer.as_mut() else {
        return 0;
    };
    if !replace_bytes(&mut state.active, bytes) {
        return 0;
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_set_ledger(
    buffer: *mut HermesStreamBuffer,
    data: *const u8,
    len: usize,
) -> i32 {
    let Some(bytes) = input_slice(data, len) else {
        return 0;
    };
    let Some(state) = buffer.as_mut() else {
        return 0;
    };
    if !replace_bytes(&mut state.ledger, bytes) {
        return 0;
    }
    1
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_clear(buffer: *mut HermesStreamBuffer) {
    if let Some(state) = buffer.as_mut() {
        state.active.clear();
        state.ledger.clear();
    }
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_active_len(
    buffer: *const HermesStreamBuffer,
) -> usize {
    buffer.as_ref().map_or(0, |state| state.active.len())
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_ledger_len(
    buffer: *const HermesStreamBuffer,
) -> usize {
    buffer.as_ref().map_or(0, |state| state.ledger.len())
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_active_copy(
    buffer: *const HermesStreamBuffer,
    out: *mut u8,
    capacity: usize,
) -> usize {
    buffer
        .as_ref()
        .map_or(0, |state| output_copy(&state.active, out, capacity))
}

#[no_mangle]
pub unsafe extern "C" fn hermes_stream_buffer_ledger_copy(
    buffer: *const HermesStreamBuffer,
    out: *mut u8,
    capacity: usize,
) -> usize {
    buffer
        .as_ref()
        .map_or(0, |state| output_copy(&state.ledger, out, capacity))
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe fn read_buffer(
        copy: unsafe extern "C" fn(*const HermesStreamBuffer, *mut u8, usize) -> usize,
        state: *const HermesStreamBuffer,
    ) -> Vec<u8> {
        let len = copy(state, ptr::null_mut(), 0);
        let mut bytes = vec![0_u8; len];
        copy(state, bytes.as_mut_ptr(), bytes.len());
        bytes
    }

    #[test]
    fn append_keeps_active_and_full_ledger() {
        unsafe {
            let state = hermes_stream_buffer_new();
            assert_eq!(hermes_stream_buffer_append(state, b"a".as_ptr(), 1), 1);
            assert_eq!(hermes_stream_buffer_append(state, b"b".as_ptr(), 1), 1);
            assert_eq!(read_buffer(hermes_stream_buffer_active_copy, state), b"ab");
            assert_eq!(read_buffer(hermes_stream_buffer_ledger_copy, state), b"ab");
            hermes_stream_buffer_free(state);
        }
    }

    #[test]
    fn replacing_active_does_not_rewrite_ledger() {
        unsafe {
            let state = hermes_stream_buffer_new();
            hermes_stream_buffer_append(state, b"full".as_ptr(), 4);
            hermes_stream_buffer_set_active(state, b"tail".as_ptr(), 4);
            assert_eq!(
                read_buffer(hermes_stream_buffer_active_copy, state),
                b"tail"
            );
            assert_eq!(
                read_buffer(hermes_stream_buffer_ledger_copy, state),
                b"full"
            );
            hermes_stream_buffer_free(state);
        }
    }
}
