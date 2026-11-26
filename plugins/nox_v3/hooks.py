"""
NOX V3 Hooks - Pre and Post LLM Call Processing
"""

import time
from typing import Dict, Any, Optional, Tuple
from . import (
    is_enabled,
    get_mode,
    get_latency_budget,
    get_fast_path_threshold,
    can_use_tokens,
    record_token_usage,
    record_verification_result,
    get_config,
)


def get_nox_system_prompt(mode: str = "balanced") -> str:
    """
    Generate NOX system prompt for compact internal reasoning.

    Args:
        mode: Compression mode (conservative/balanced/aggressive)

    Returns:
        System prompt string
    """
    base_prompt = """You are a NOX compiler/decoder. Follow these rules:

1. **Compact Internal Reasoning**: Think and reason using concise NOX notation internally.
   Use short keywords/symbols instead of verbose phrases.
   Only include essential facts and logical steps.
   Minimize token use while preserving all input information.

2. **Grammar/Structure**: Use strict NOX notation:
   - FACT[condition] for facts
   - RULE[X->Y] for rules
   - INFER[conclusion] for inferences
   - VERIFY[step] for verification
   Each line should represent one atomic operation.

3. **Verify & Preserve**: After producing NOX encoding, check that it covers
   ALL user-provided premises and questions. Do NOT invent new facts.
   Mark uncertain parts with REVIEW: tag.

4. **Expand Answer**: Output the final answer in clear natural language.
   Do NOT output the NOX steps themselves—only the final reasoning conclusion.
"""

    mode_specific = {
        "conservative": """
5. **Conservative Mode**: Keep terms readable. Use minor abbreviation.
   Example: IF (Wed is between Mon-Fri) THEN open.
""",
        "balanced": """
5. **Balanced Mode**: Use symbolic logic with moderate compression.
   Example: X→Y; Y→Z.
""",
        "aggressive": """
5. **Aggressive Mode**: Use maximum shorthand and symbols.
   Example: X->Y,Y->Z.
""",
    }

    return base_prompt + mode_specific.get(mode, mode_specific["balanced"])


def estimate_tokens(text: str) -> int:
    """
    Estimate token count (rough approximation).

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    # Rough approximation: ~4 characters per token
    return len(text) // 4


def should_use_fast_path(response: str) -> bool:
    """
    Determine if response should use fast path (skip verification).

    Args:
        response: LLM response

    Returns:
        True if fast path should be used
    """
    token_count = estimate_tokens(response)
    threshold = get_fast_path_threshold()
    return token_count < threshold


def pre_llm_call(ctx, config: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
    """
    Pre-LLM call hook - inject NOX system prompt.

    Args:
        ctx: Hermes context
        config: Plugin configuration
        **kwargs: Additional arguments

    Returns:
        Dictionary with context modifications, or None
    """
    # Check if NOX is enabled
    if not is_enabled():
        return None

    # Check token budget
    nox_prompt = get_nox_system_prompt(get_mode())
    nox_tokens = estimate_tokens(nox_prompt)

    if not can_use_tokens(nox_tokens):
        ctx.log("NOX V3: Token budget exceeded, skipping")
        return None

    # Record token usage for system prompt
    record_token_usage(nox_tokens)

    # Store metadata for post-processing
    ctx.set_state("nox_v3", {
        "enabled": True,
        "mode": get_mode(),
        "start_time": time.time(),
        "latency_budget": get_latency_budget(),
        "nox_prompt_tokens": nox_tokens,
    })

    # Inject NOX system prompt
    ctx.log(f"NOX V3: Injecting system prompt ({nox_tokens} tokens, mode={get_mode()})")

    return {
        "context": nox_prompt,
    }


def parse_nox_reasoning(response: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse NOX reasoning from response.

    Args:
        response: LLM response

    Returns:
        Tuple of (nox_reasoning, final_answer)
    """
    # Try to extract NOX reasoning (if present in output)
    # Most models should follow instructions and only output final answer
    # But we handle cases where NOX reasoning is visible

    lines = response.split("\n")
    nox_lines = []
    final_lines = []
    in_nox = False

    for line in lines:
        # Detect NOX notation patterns
        if any(pattern in line for pattern in ["FACT[", "RULE[", "INFER[", "VERIFY["]):
            in_nox = True
            nox_lines.append(line)
        elif in_nox and line.strip():
            # Continue collecting NOX lines
            nox_lines.append(line)
        else:
            # Final answer
            final_lines.append(line)

    nox_reasoning = "\n".join(nox_lines) if nox_lines else None
    final_answer = "\n".join(final_lines).strip() if final_lines else response.strip()

    return nox_reasoning, final_answer


def verify_reasoning(nox_reasoning: str, original_query: str) -> Tuple[bool, str]:
    """
    Verify NOX reasoning for completeness and correctness.

    Args:
        nox_reasoning: NOX reasoning to verify
        original_query: Original user query

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Lightweight verification checks

    # Check 1: Non-empty reasoning
    if not nox_reasoning or not nox_reasoning.strip():
        return False, "Empty reasoning"

    # Check 2: Contains logical structure
    has_structure = any(
        pattern in nox_reasoning
        for pattern in ["FACT[", "RULE[", "INFER[", "VERIFY[", "->", "→"]
    )
    if not has_structure:
        return False, "Missing logical structure"

    # Check 3: No REVIEW tags (indicates uncertainty)
    if "REVIEW:" in nox_reasoning:
        return False, "Uncertain reasoning marked for review"

    # Check 4: Reasoning covers query (basic keyword matching)
    query_words = set(original_query.lower().split())
    reasoning_words = set(nox_reasoning.lower().split())
    overlap = len(query_words & reasoning_words)

    # At least 30% of query words should appear in reasoning
    if len(query_words) > 0 and overlap / len(query_words) < 0.3:
        return False, f"Insufficient query coverage ({overlap}/{len(query_words)} words)"

    return True, ""


def optimize_reasoning(nox_reasoning: str, mode: str) -> str:
    """
    Optimize NOX reasoning based on mode.

    Args:
        nox_reasoning: NOX reasoning to optimize
        mode: Compression mode

    Returns:
        Optimized reasoning
    """
    # Apply mode-specific optimizations

    if mode == "conservative":
        # Minimal optimization - just clean up whitespace
        return "\n".join(line.strip() for line in nox_reasoning.split("\n") if line.strip())

    elif mode == "balanced":
        # Moderate optimization - remove redundant words
        lines = []
        for line in nox_reasoning.split("\n"):
            if line.strip():
                # Remove common filler words
                for filler in ["the ", "a ", "an ", "is ", "are ", "was ", "were "]:
                    line = line.replace(filler, " ")
                lines.append(line.strip())
        return "\n".join(lines)

    elif mode == "aggressive":
        # Maximum optimization - use symbols
        lines = []
        for line in nox_reasoning.split("\n"):
            if line.strip():
                # Replace common patterns with symbols
                line = line.replace("implies", "->")
                line = line.replace("therefore", "→")
                line = line.replace("because", "∵")
                line = line.replace("thus", "∴")
                lines.append(line.strip())
        return "\n".join(lines)

    return nox_reasoning


def post_llm_call(ctx, response: str, config: Dict[str, Any], **kwargs) -> Optional[str]:
    """
    Post-LLM call hook - verify and optimize response.

    Args:
        ctx: Hermes context
        response: LLM response
        config: Plugin configuration
        **kwargs: Additional arguments

    Returns:
        Optimized response, or None to use original
    """
    # Get NOX state from pre-hook
    nox_state = ctx.get_state("nox_v3")
    if not nox_state or not nox_state.get("enabled"):
        return None

    # Check latency budget
    elapsed_ms = (time.time() - nox_state["start_time"]) * 1000
    latency_budget = nox_state["latency_budget"]

    if elapsed_ms > latency_budget:
        ctx.log(f"NOX V3: Latency budget exceeded ({elapsed_ms:.1f}ms > {latency_budget}ms), using original")
        return None

    # Check fast path
    if should_use_fast_path(response):
        ctx.log("NOX V3: Fast path (simple response), skipping verification")
        return None

    # Parse response
    nox_reasoning, final_answer = parse_nox_reasoning(response)

    # If no NOX reasoning found, model didn't follow instructions
    # Just return final answer as-is
    if not nox_reasoning:
        ctx.log("NOX V3: No NOX reasoning found, using final answer")
        return final_answer if final_answer else None

    # Verify reasoning
    original_query = kwargs.get("query", "")
    is_valid, error_msg = verify_reasoning(nox_reasoning, original_query)

    if not is_valid:
        ctx.log(f"NOX V3: Verification failed ({error_msg}), using original")
        record_verification_result(False)
        return None

    # Optimize reasoning
    mode = nox_state["mode"]
    optimized_reasoning = optimize_reasoning(nox_reasoning, mode)

    # Record token usage for optimization
    opt_tokens = estimate_tokens(optimized_reasoning)
    record_token_usage(opt_tokens)

    # Record success
    record_verification_result(True)

    ctx.log(f"NOX V3: Verification passed, optimized ({opt_tokens} tokens)")

    # Return final answer (NOX reasoning is internal)
    return final_answer if final_answer else response
