# Hermes troubleshooting

- Run `hermes mcp list` and `hermes mcp test Plugin-Hermes-kling-ai` to inspect the native connection.
- Run `hermes mcp login Plugin-Hermes-kling-ai` for first authorization or intentional re-authentication.
- Run `/reload-mcp` after changing `~/.hermes/config.yaml`.
- On SSH or a remote gateway, use Hermes' documented redirect-URL paste flow or port forwarding. Never paste access or refresh tokens into chat.
- If a submission times out, query existing tasks by returned identifiers or current account history. Never blind-retry a credit-consuming tool.
