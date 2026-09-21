# Idea Box — Discord source triage to approval-ready Kanban tasks
#
# Subsystem: Discord Intake + Embed Presenter (gojo domain)
# Architecture: /home/kensei/.hermes/kanban/workspaces/t_d31645ef/IDEA_BOX_ARCHITECTURE.md
#
# Public API:
#   IdeaBoxDiscordHandler  — slash command + message listener + button handler
#   IdeaBoxApprovalView    — Approve/Amend/Reject Discord UI view
#   create_ideabox_embed   — Build the triage summary embed
#   handle_ideabox_submission — Full pipeline: parse → triage → embed → store
#
# Channel-level intake hook (added in t_1b051b8b):
#   Set DISCORD_IDEABOX_CHANNELS=<comma-separated channel IDs> to
#   designate text and/or forum channels where any posted link/article
#   is auto-templated through the Idea Box pipeline. Messages in those
#   channels are intercepted before the normal Hermes agent pipeline
#   and converted into triage embeds with approval buttons.
#
# Slash command:
#   /ideabox <url|article|github>  — works in any channel, ephemeral-friendly
#
# Message listener:
#   Any channel in DISCORD_IDEABOX_CHANNELS — free-form submission,
#   requires no mention prefix when the channel is in the list.
