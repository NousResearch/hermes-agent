# Group Chat Messaging Core

Messaging controls use the same durable room log and driver as Desktop. The
canonical command family is `/group`; Slack and Matrix use their receiving
transport's `!` prefix. Private controls require an authorized person. A shared
Home requires the exact enrolled selector, location, topic, scope, and audience
acknowledgement. `/sethome` chooses a destination; it never enrolls an owner.

## Owners

- `gateway/group_chat_slash.py`: command parsing, authorization, menus, and bounded
  reads/mutations; installed through `GatewaySlashCommandsMixin`.
- `gateway/group_chat_policy.py`: live receiving-adapter policy, including routed
  profiles and fail-closed retained transport references.
- `gateway/group_home_identity.py`, `group_home_consent.py`, and
  `home_channel_config.py`: private-audience proof, consent lifecycle, and exact
  Home replacement. Existing `gateway/config_io.py` owns persisted Home binding
  restoration and writes; this port does not duplicate that foundation.
- `gateway/hosted_room_messaging.py`: bounded room selection/presentation, trusted
  actor/event identity, and hosted, reciprocal, or classic-room control routing.
- `gateway/hosted_room_controls.py` and `hosted_room_control_client.py`: private
  reciprocal credentials, exact room authority, revocation, and durable commands.
- `gateway/hosted_room_messaging_approvals.py` and
  `hosted_room_messaging_retries.py`: exact-request approval decisions and durable
  retry receipts. A response-lost replay cannot retarget newer work.
- `gateway/desktop_room_mailbox.py`: classic-room commands leased to a Desktop
  consumer that proves its room authority; queued is not executed.
- `gateway/hosted_room_driver.py`: active-lease checks, retry migration/receipts,
  and stop fences before admission or explicit requeue.
- `tui_gateway/hosted_room_service.py` and `hosted_room_driver.py`: restore pending
  observations, apply controls under the live lease, and publish trusted sends.
- `tui_gateway/methods_groups.py`: `groups.desktop.{claim,presence,renew,complete}`
  and `groups.control.{invite,register,revoke}`, all on the RPC worker pool.
- `gateway/platforms/api_server_room_controls.py`: scoped reciprocal control HTTP
  routes, registered by `api_server_room_grants.py`.
- `tui_gateway/change_watcher.py`: `desktop_rooms.commands.pending` notification.
- `gateway/run_busy.py` and `hermes_cli/commands.py`: shared idle/busy dispatch and
  command policy. `gateway/session.py` retains author classification but does not
  serialize live privacy, edit, or attachment provenance.

## Native Integration Contract

`GatewayModelCommandsMixin._try_send_choice_picker` passes the original callback,
session key, requester, reply anchor, and topic to the receiving adapter. Reusable
pages require the adapter type's `supports_choice_pages is True` and a requester.
Unsupported adapters keep the text fallback. A callback may return `ChoicePage`
or `ChoiceProgress`; native renderers own exact prompt identity, expiry, revision,
single entry, progress-before-work, and retirement.

Only recognized bundled Telegram/Discord picker owners may render Home consent.
Core callback tests do not establish native SDK rendering or lifecycle correctness.
Native modules and their SDK tests are a separate integration unit.

## Boundaries

New send/retry work is refused during maintenance. Stop and deny remain available
to an authorized owner. Started workers stay visible to maintenance even after
their caller disconnects. Every asynchronous read and callback rechecks the
captured disclosure scope before publishing private output or starting more work.

Retry of an indeterminate attempt can repeat actions. Replaying an accepted retry
receipt does not create a new attempt. Stop requests fence older tasks without
claiming a remote process has already stopped.

Files, blob transport, Bot-output publication, and Desktop TypeScript are not part
of this core. The lock-free CLI config snapshot hook is supplied by the separate
config integration; it is needed for emergency control while a Home save is held.

## Provenance

Ported from David Dudok de Wit's tested Messaging source at
`b77d4505ba1ba0d6b05aba199f740f05c0e9b85d`, beginning with
`9b1c178f82` after `9954445fea2e3c85034d1afc6ebd40abdaa91f71`, including the
subsequent control, authorization, consent, maintenance, and retry repairs. The
port retains current main's split owners and grant-refresh behavior instead of
restoring the former monolithic gateway or server modules.
