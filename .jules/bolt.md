## 2024-05-24 - Deepcopy on message histories
**Learning:** `copy.deepcopy` is extremely slow when applied to conversation history (which can contain huge text payloads).
**Action:** When sanitizing or cloning message lists before sending to transports, avoid `deepcopy` and use manual shallow copying (`dict.copy()` and list comprehensions) on just the structures that need modification.
