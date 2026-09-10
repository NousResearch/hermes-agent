# 2Chat MCP — Tool Reference

All tools become available once the `2chat` MCP server is connected (see SKILL.md → Quick Reference).

## Account
- **get_who_am_i** — Return information about the authenticated 2Chat account.
- **get_billing_info** — Current billing status, plan limits, and API usage metrics.

## Messaging — WhatsApp Web
- **send_whatsapp_message** — Send a text or media message via a connected WhatsApp Web channel.
- **check_if_number_is_on_whatsapp** — Verify whether a phone number has an active WhatsApp account.
- **get_whatsapp_messages** — Retrieve WhatsApp messages exchanged through a connected number.

## Messaging — WhatsApp Business API (WABA)
- **send_waba_message** — Send a WABA message using an approved template or free-form text.
- **get_waba_templates** — List WABA message templates (optional filtering).
- **sync_waba_templates** — Trigger a background sync of WABA templates from Meta.
- **calculate_waba_template_cost** — Estimate messaging cost by plan, country, and template type.

## Messaging — SMS
- **send_sms** — Send an SMS through a 2Chat SMS channel.

## Channels — WhatsApp Web
- **get_whatsapp_numbers** — List WhatsApp Web numbers connected via QR code.
- **get_whatsapp_number** — Full details for a single WhatsApp Web channel.
- **execute_whatsapp_channel_command** — Connect/disconnect operations on a WhatsApp channel.

## Channels — WABA
- **get_waba_numbers** — List WABA numbers connected to the account.
- **get_waba_number** — Full details of a single WABA number.

## Conversations & Groups
- **list_whatsapp_conversations** — List WhatsApp conversations ordered by recent activity.
- **list_whatsapp_groups** — List WhatsApp groups visible to a connected number.
- **list_whatsapp_group_participants** — List participants of a WhatsApp group.
- **get_whatsapp_group_messages** — Retrieve messages from a WhatsApp group.

## Status (Stories)
- **set_whatsapp_text_status** — Publish an ephemeral text status.
- **set_whatsapp_image_status** — Publish an ephemeral image status.
- **set_whatsapp_video_status** — Publish an ephemeral video status.

## Catalog
- **list_whatsapp_catalog_products** — List products from WhatsApp Business catalogs.

## Contacts
- **create_contact** — Create a new contact.
- **get_contact** — Retrieve a single contact by UUID.
- **list_contacts** — List contacts (optional channel filtering).
- **search_contacts** — Find contacts by name or phone number.
- **update_contact** — Modify an existing contact.
- **delete_contact** — Permanently remove a contact (irreversible).

## Calls & Virtual Numbers
- **list_virtual_numbers** — List VoIP numbers connected to the account.
- **get_call_history** — Get call detail records (CDRs).
- **get_call_details** — Full details for a specific call.
- **get_call_price** — Estimate per-minute outbound call cost to a destination.

---
Docs: https://developers.2chat.co/docs/MCP/setup
