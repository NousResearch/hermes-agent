# Live Glass — Manual Test Plan

Run these checks before marking the PR ready for review.

## Prerequisites

- Hermes Agent running on macOS with cua-driver installed
- Dashboard accessible at http://localhost:18724 (or configured port)
- Telegram gateway running (if testing chat adapters)

## 1. Plugin loading

- [ ] `hermes plugins enable observability/live-glass`
- [ ] Restart Hermes (gateway or CLI)
- [ ] Confirm plugin loads: `hermes plugins list | grep live-glass`

## 2. Event bus in isolation

- [ ] Start a Hermes CLI session
- [ ] Trigger a `computer_use` capture action
- [ ] Verify the live-glass plugin's `on_post_tool_call` fires (check agent.log for live-glass debug entries)
- [ ] Trigger a dangerous shell command (e.g. `rm -rf /tmp/test-dir`) to fire approval hooks
- [ ] Verify `pre_approval_request` and `post_approval_response` events are emitted

## 3. Dashboard WebSocket stream

- [ ] Start dashboard: `hermes dashboard`
- [ ] Open browser devtools → Network → WS tab
- [ ] Connect to `ws://localhost:18724/api/plugins/live-glass/events`
- [ ] In a separate Hermes session, run `computer_use` actions
- [ ] Verify frame events arrive as JSON with `type: "frame"`
- [ ] Verify log events arrive with `type: "log"`
- [ ] Verify heartbeat messages arrive every ~30s with `type: "heartbeat"`
- [ ] Verify last frame is replayed on reconnect
- [ ] Verify multiple browser tabs can connect simultaneously

## 4. Dashboard Live view tab

- [ ] Navigate to dashboard → look for "Live Glass" tab (hidden by default in prototype)
- [ ] Or manually navigate to `/live-glass` route if registered
- [ ] Verify viewport shows the latest frame screenshot
- [ ] Verify "Waiting for computer_use activity..." placeholder when no frame
- [ ] Verify log section scrolls with events
- [ ] Verify approval controls appear when an approval_request is received
- [ ] Verify connection status dot shows green when WebSocket is live

## 5. Telegram adapter (if gateway is running)

- [ ] Send a message to the Hermes Telegram bot to start a session
- [ ] Trigger a `computer_use` capture in that session
- [ ] Verify a photo message arrives in Telegram with the screenshot
- [ ] Trigger a dangerous action — verify inline keyboard with Approve/Deny buttons
- [ ] Click "Approve Once" — verify the action proceeds
- [ ] Click "Deny" — verify the action is blocked

## 6. Approval bridge

- [ ] Trigger a `computer_use click` action (destructive, requires approval)
- [ ] Verify the approval prompt appears (CLI or gateway depending on surface)
- [ ] Verify the live-glass event bus received `approval_request` event
- [ ] Approve the action
- [ ] Verify `post_approval_response` event fires with `choice: "approve_once"`
- [ ] Deny another action — verify `choice: "deny"`

## 7. Frame poller (optional)

- [ ] In a Python shell: `from plugins.observability.live_glass.frame_poller import FramePoller, computer_use_backend_factory`
- [ ] `backend = computer_use_backend_factory()`
- [ ] `poller = FramePoller(backend, interval=2.0)`
- [ ] `poller.start()`
- [ ] Wait 5 seconds, then check `get_events(event_type="frame")` — should have polled frames
- [ ] `poller.stop()`

## 8. Error handling

- [ ] Stop cua-driver while poller is running — verify poller survives and continues on backend restoration
- [ ] Disconnect WebSocket client mid-stream — verify server cleans up without crash
- [ ] Send an approval_request with an unmapped session_id — verify no message sent, no crash

## 9. Cross-platform

- [ ] Run `scripts/check-windows-footguns.py plugins/observability/live_glass/` — verify clean
- [ ] Confirm no hardcoded macOS paths or assumptions outside the computer_use backend adapter
