# Hermes Agent ACP launcher

This is the small `npx` launcher used by the ACP Registry entry for Hermes
Agent. It starts the user's installed `hermes-acp` process, preserving the
normal Hermes configuration, credentials, skills, memory, and model catalog.

Install Hermes first with the official installer:

```sh
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

The launcher prefers the managed `hermes-acp` executable, then a launcher on
`PATH`, and finally falls back to `uvx --from hermes-agent[acp] hermes-acp` on
POSIX systems where `uv` is already installed.
