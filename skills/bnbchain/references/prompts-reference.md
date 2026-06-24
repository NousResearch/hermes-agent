# MCP prompts reference

The BNB Chain MCP server exposes **prompts** that return guided analysis or explanations. Use these prompt names when invoking MCP prompts (e.g. from Cursor or Claude).

| Prompt name | When to use |
|-------------|-------------|
| **analyze_block** | User wants detailed information about a block (transactions, gas, etc.). |
| **analyze_transaction** | User wants analysis of a specific transaction (hash). |
| **analyze_address** | User wants analysis of an EVM address (balances, contracts, activity). |
| **interact_with_contract** | User needs guidance on how to interact with a smart contract (read/write, ABI, params). |
| **explain_evm_concept** | User asks about an EVM concept (gas, opcodes, ABI, etc.). |
| **compare_networks** | User wants to compare EVM-compatible networks (BSC, opBNB, Ethereum, etc.). |
| **analyze_token** | User wants to analyze an ERC20 or NFT token (metadata, supply, etc.). |
| **how_to_register_mcp_as_erc8004_agent** | User wants step-by-step guidance on registering an MCP server as an ERC-8004 agent. |

Prompts typically accept input (e.g. block number, tx hash, address) as specified by the MCP server. Prefer these prompts when the user asks for “analysis,” “explanation,” or “how to” rather than a single raw tool call.
