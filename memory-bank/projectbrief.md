# Project Brief: Hermes-Agent

## Overview
Hermes-Agent is an AI agent harness for LLMs with advanced tool-calling capabilities, featuring a flexible toolsets system for organizing and managing tools. Named after Hermes, the Greek messenger god, it serves as a bridge between human intent and AI-powered task execution.

## Core Requirements

### Primary Goals
1. **Interactive CLI Experience** - Beautiful terminal interface with animated feedback, personalities, and session management
2. **Flexible Tool System** - Modular tools organized into logical toolsets for different use cases
3. **Batch Processing** - Process multiple prompts in parallel with checkpointing and statistics
4. **Multi-Backend Support** - Support for local, Docker, Singularity, Modal, and SSH terminal backends
5. **Training Data Generation** - Save conversation trajectories in formats suitable for LLM fine-tuning

### Target Users
- AI researchers generating training data
- Developers needing an AI assistant with tool access
- MLOps practitioners automating workflows
- Anyone needing a powerful CLI-based AI agent

## Scope

### In Scope
- Interactive CLI with rich formatting and kawaii-style feedback
- Web tools (search, extract, crawl via Firecrawl)
- Terminal tools (command execution across multiple backends)
- Browser automation (via agent-browser + Browserbase)
- Vision tools (image analysis)
- Image generation (FLUX via FAL.ai)
- Mixture-of-Agents reasoning
- Skills system for on-demand knowledge
- Batch processing with parallel workers
- Trajectory compression for training

### Out of Scope (Current)
- Proactive suggestions (agent only runs on request)
- Clipboard integration (no local system access)
- Real-time streaming of thinking/reasoning (deferred)

## Success Metrics
- Clean, maintainable tool architecture
- Reliable tool execution with proper error handling
- Efficient context management for long conversations
- High-quality trajectory data for training
