# Bot conversations in Desktop

An expanded bot row shows that bot's ordinary, visible sessions beneath the canonical **Bot Chat**. These are independent session histories, not aliases of the canonical chat. The bot row still opens the canonical chat; selecting a child opens its stored session ID as a tab. The list reads the bot's source and backend profile, shows the five most recent sessions first, and places the rest behind **Earlier conversations**. The child age is a compact relative time.

## Deliberate context-menu limitation

**Right-click is disabled on the nested conversation list.** The bot profile's context-menu trigger wraps the expanded list; letting a child right-click propagate exposes the _bot profile_ menu, whose **Delete** action deletes the bot profile rather than the selected conversation. The list prevents the default context menu and stops propagation. Left-click and keyboard activation of a child still open the conversation normally, and right-click on the bot row still opens the bot profile menu.

Future contributors may implement a dedicated child-session context menu using the existing session actions, with explicit session-ID/source/profile ownership and correct archive/delete confirmation. Do not simply re-enable propagation to the parent's menu, and do not mistake the profile Delete action for a child-session action. This contribution intentionally does not implement child deletion or rename from the list; ordinary session management remains available elsewhere in Desktop.

See [issue #112184](https://github.com/NousResearch/hermes-agent/issues/112184) and [Finn763's original PR #112688](https://github.com/NousResearch/hermes-agent/pull/112688) for the initial discovery/navigation proposal.
