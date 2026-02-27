"""Agent internals -- extracted modules from run_agent.py.

These modules contain pure utility functions and self-contained classes
that were previously embedded in the 3,600-line run_agent.py. Extracting
them makes run_agent.py focused on the AIAgent orchestrator class.

Module Overview
---------------
This package contains the following components:

**prompt_builder.py**
    System prompt assembly -- identity, platform hints, skills index,
    and context file loading. All functions are stateless and used by
    AIAgent._build_system_prompt() to assemble the system prompt pieces.

**context_compressor.py**
    Automatic context window compression for long conversations. Uses
    a summarization model to compress middle turns while protecting
    head (system prompt) and tail (recent context).

**auxiliary_client.py**
    Shared OpenAI client for cheap/fast side tasks like summarization,
    vision analysis, and web extraction. Provides a single resolution
    chain (OpenRouter -> Nous Portal -> custom endpoint).

**model_metadata.py**
    Model metadata, context lengths, and token estimation utilities.
    Fetches metadata from OpenRouter API with caching.

**prompt_caching.py**
    Anthropic prompt caching (system_and_3 strategy) to reduce input
    token costs by ~75% on multi-turn conversations.

**display.py**
    CLI presentation -- spinner, kawaii faces, and tool preview formatting.
    Pure display functions with no AIAgent dependency.

**trajectory.py**
    Trajectory saving utilities for exporting conversations in ShareGPT
    format for training data generation.

Architecture
------------
The agent package follows these design principles:

1. **Stateless utilities**: Most functions are pure and take all needed
   state as arguments, making them easy to test and reuse.

2. **No circular imports**: Modules only depend on external packages and
   hermes_constants, never on run_agent.py or each other's classes.

3. **Single responsibility**: Each module handles one aspect of agent
   functionality (prompt building, compression, caching, display).

4. **AIAgent as orchestrator**: The main AIAgent class in run_agent.py
   coordinates these modules but doesn't contain their implementation.
"""
