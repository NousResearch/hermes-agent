# Prompt warmer

Opt-in prompt warmer: primes the server's prompt cache with the exact
session preamble at session open, so the first user turn processes
only the new tokens. Default off. The warm request starts the model
on llama-swap exactly like a real first turn would - that is the
point, but it is why the gate is strict (boolean true only).

## Config surface

    providers.llamacpp.models.<id>.prompt_warmer   boolean, wins over agent.prompt_warmer
    agent.prompt_warmer                            opt-in prompt warmer
