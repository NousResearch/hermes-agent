use pyo3::prelude::*;
use pyo3::types::PyList;

/// Rough token estimate (~4 chars/token) for pre-flight checks.
#[pyfunction]
pub fn estimate_tokens_rough(text: &str) -> usize {
    text.len() / 4
}

/// Rough token estimate for a message list.
/// Calls str() on each element, sums lengths, divides by 4.
#[pyfunction]
pub fn estimate_messages_tokens_rough(messages: &Bound<'_, PyList>) -> PyResult<usize> {
    let mut total_bytes: usize = 0;
    for item in messages.iter() {
        let s = item.str()?;
        total_bytes += s.len()?;
    }
    Ok(total_bytes / 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_returns_zero() {
        assert_eq!(estimate_tokens_rough(""), 0);
    }

    #[test]
    fn four_chars_is_one_token() {
        assert_eq!(estimate_tokens_rough("abcd"), 1);
    }
}
