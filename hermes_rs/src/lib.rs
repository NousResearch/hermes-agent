use pyo3::prelude::*;

mod agent_loop;
mod prompt_scanner;
mod token_estimate;

#[pymodule]
fn hermes_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Prompt scanner
    m.add_function(wrap_pyfunction!(prompt_scanner::scan_context_content, m)?)?;
    m.add_function(wrap_pyfunction!(prompt_scanner::truncate_content, m)?)?;

    // Token estimation
    m.add_function(wrap_pyfunction!(token_estimate::estimate_tokens_rough, m)?)?;
    m.add_function(wrap_pyfunction!(token_estimate::estimate_messages_tokens_rough, m)?)?;

    // Agent loop state machine
    m.add_class::<agent_loop::LoopState>()?;
    m.add_class::<agent_loop::Action>()?;
    m.add_class::<agent_loop::ResponseKind>()?;
    m.add_class::<agent_loop::Transition>()?;
    m.add_class::<agent_loop::AgentLoopMachine>()?;
    m.add_function(wrap_pyfunction!(agent_loop::strip_think_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(agent_loop::strip_tool_call_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(agent_loop::has_content_after_think, m)?)?;

    Ok(())
}
