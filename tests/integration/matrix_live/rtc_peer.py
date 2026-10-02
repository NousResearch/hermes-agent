"""Independent Matrix and LiveKit participant used by the Linux RTC proof."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from urllib.parse import quote

import aiohttp
from livekit import rtc
from nio import RoomMessagesResponse

from client import open_encrypted_client
from rtc_gateway import audio_summary, tone

MEMBER_TYPE = "org.matrix.msc3401.call.member"
SERVICE_URL = "http://rtc-auth:8080"


async def wait_until(check, description: str, timeout: float = 25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = await check()
        if result:
            return result
        await asyncio.sleep(0.1)
    trace = Path("/opt/data/rtc-peer-state.json")
    details = trace.read_text() if trace.exists() else "unavailable"
    raise AssertionError(f"Timed out waiting for {description}; bot state: {details}")


async def credentials(http, user_id, access_token, device_id, room_id):
    headers = {"Authorization": f"Bearer {access_token}"}
    async with http.post(f"http://synapse:8008/_matrix/client/v3/user/{quote(user_id, safe='')}/openid/request_token",
                         headers=headers, json={}) as response:
        assert response.status == 200, response.status
        openid = await response.json()
    async with http.post("http://rtc-auth:8080/sfu/get", json={
            "room": room_id, "openid_token": {**openid, "access_token": "invalid-token"},
            "device_id": device_id}) as response:
        assert response.status in (401, 403), response.status
    async with http.post("http://rtc-auth:8080/sfu/get", json={
            "room": room_id, "openid_token": openid, "device_id": device_id}) as response:
        assert response.status == 200, response.status
        body = await response.json()
    claims = json.loads(base64.urlsafe_b64decode(body["jwt"].split(".")[1] + "=="))
    assert claims["sub"] == f"{user_id}:{device_id}"
    return body["url"], body["jwt"]


async def publish(room, frequency: int):
    source = rtc.AudioSource(48000, 1)
    track = rtc.LocalAudioTrack.create_audio_track("independent-peer", source)
    publication = await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    try:
        pcm = tone(frequency, 48000, 0.9) + bytes(48000 * 2)
        for offset in range(0, len(pcm), 1920):
            chunk = pcm[offset:offset + 1920]
            await source.capture_frame(rtc.AudioFrame(chunk, 48000, 1, len(chunk) // 2))
        await source.wait_for_playout()
    finally:
        await room.local_participant.unpublish_track(publication.sid)
        await source.aclose()


async def run(room_id: str, bot_id: str, mode: str, mallory: dict, startup_grace: float):
    matrix = open_encrypted_client()
    room = rtc.Room()
    tasks = set()
    received = bytearray()
    decoded = {}
    heard = asyncio.Event()
    subscribed = asyncio.Event()
    paths = Path("/opt/data")
    try:
        await matrix.sync(timeout=1000)
        powers = {"users": {matrix.user_id: 100, bot_id: 50, mallory["user_id"]: 50},
                  "users_default": 0, "events_default": 0, "state_default": 50,
                  "ban": 50, "kick": 50, "redact": 50, "invite": 0}
        result = await matrix.room_put_state(room_id, "m.room.power_levels", powers)
        assert hasattr(result, "event_id"), result
        key = f"_{matrix.user_id}_{matrix.device_id}_m.call"
        content = {"application": "m.call", "call_id": "", "device_id": matrix.device_id,
                   "expires": 14_400_000, "scope": "m.room",
                   "focus_active": {"type": "livekit", "focus_selection": "multi_sfu"},
                   "foci_preferred": [{"type": "livekit", "livekit_alias": room_id,
                                       "livekit_service_url": SERVICE_URL}]}
        result = await matrix.room_put_state(room_id, MEMBER_TYPE, content, key)
        assert hasattr(result, "event_id"), result
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as http:
            url, jwt = await credentials(http, matrix.user_id, matrix.access_token, matrix.device_id, room_id)

            async def drain(track):
                stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1, capacity=50)
                try:
                    async for event in stream:
                        received.extend(bytes(event.frame.data))
                        summary = audio_summary(received, 48000)
                        if (not heard.is_set() and summary["duration"] >= 0.5 and summary["rms"] > 500
                                and abs(summary["frequency"] - 880) < 12):
                            decoded.update(summary)
                            heard.set()
                finally:
                    await stream.aclose()

            @room.on("track_subscribed")
            def on_track(track, publication, participant):
                if participant.identity.startswith(bot_id + ":") and track.kind == rtc.TrackKind.KIND_AUDIO:
                    subscribed.set()
                    task = asyncio.create_task(drain(track))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)

            await room.connect(url, jwt)
            async def body_seen(body):
                response = await matrix.room_messages(room_id, limit=40)
                assert isinstance(response, RoomMessagesResponse), response
                return any(getattr(event, "sender", None) == bot_id and body in getattr(event, "body", "")
                           for event in response.chunk)

            if mode == "leave":
                await matrix.room_send(room_id, "m.room.message", {"msgtype": "m.text", "body": "Remember this typed question."})
                await wait_until(lambda: body_seen("Typed context reply"), "typed gateway reply")
            await matrix.room_send(room_id, "m.room.message", {"msgtype": "m.text", "body": "/voice join"})
            join_sent = time.time()
            await asyncio.wait_for(subscribed.wait(), 25)
            bot_key = f"_{bot_id}_"
            async def bot_membership_event():
                async with http.get(f"http://synapse:8008/_matrix/client/v3/rooms/{quote(room_id, safe='')}/state",
                                    headers={"Authorization": f"Bearer {matrix.access_token}"}) as response:
                    assert response.status == 200, response.status
                    events = await response.json()
                matches = [event for event in events if event.get("type") == MEMBER_TYPE
                           and event.get("state_key", "").startswith(bot_key)]
                (paths / "rtc-peer-state.json").write_text(json.dumps(matches))
                return matches[0] if matches else None

            async def bot_membership():
                event = await bot_membership_event()
                return event["content"] if event else None

            joined = await wait_until(bot_membership, "client-visible bot call membership")
            assert {key: joined.get(key) for key in ("application", "call_id", "focus_active")} == {
                "application": "m.call", "call_id": "",
                "focus_active": {"type": "livekit", "focus_selection": "multi_sfu"}}, joined
            assert joined["foci_preferred"][0]["livekit_service_url"] == SERVICE_URL, joined
            if mode == "leave":
                other = rtc.Room()
                try:
                    mallory_key = f"_{mallory['user_id']}_{mallory['device_id']}_m.call"
                    async with http.put(f"http://synapse:8008/_matrix/client/v3/rooms/{quote(room_id, safe='')}/state/{MEMBER_TYPE}/{quote(mallory_key, safe='')}",
                                        headers={"Authorization": f"Bearer {mallory['access_token']}"},
                                        json={**content, "device_id": mallory["device_id"]}) as response:
                        assert response.status == 200, response.status
                    other_url, other_jwt = await credentials(http, mallory["user_id"], mallory["access_token"], mallory["device_id"], room_id)
                    await other.connect(other_url, other_jwt)
                    await publish(other, 330)
                finally:
                    await other.disconnect()
                await publish(room, 660)
                await asyncio.wait_for(heard.wait(), 25)
                await wait_until(lambda: body_seen("RTC audio reply"), "spoken gateway reply")
            if mode == "restart":
                # A restarted adapter handles messages sent less than its startup grace before
                # it started, so a restart inside that window would replay this /voice join.
                # Persisted sync cursors (#126281) stop the replay and make this wait unnecessary.
                await asyncio.sleep(max(0.0, join_sent + startup_grace + 0.5 - time.time()))
            (paths / "rtc-peer-ready.json").write_text(json.dumps({"mode": mode, "membership": joined,
                                                                  "received": decoded}))
            async def control_ready():
                path = paths / "rtc-peer-control.json"
                return json.loads(path.read_text()) if path.exists() else None
            control = await wait_until(control_ready, "host lifecycle action", timeout=60)
            async def absent():
                return await bot_membership() == {}
            if control["mode"] == "leave":
                await matrix.room_send(room_id, "m.room.message", {"msgtype": "m.text", "body": "/voice leave"})
                await wait_until(lambda: body_seen("Left voice channel"), "voice leave command")
            if control["mode"] == "restart":
                # The delayed leave clears the killed gateway's membership. /voice leave in the
                # restarted gateway then has no call to leave and must not write the event again.
                await wait_until(absent, "delayed leave after the crash", timeout=25)
                cleared = (await bot_membership_event())["event_id"]
                await matrix.room_send(room_id, "m.room.message", {"msgtype": "m.text", "body": "/voice leave"})
                await wait_until(lambda: body_seen("Not in a voice channel"), "voice leave command")
                assert (await bot_membership_event())["event_id"] == cleared
            await wait_until(absent, "client-visible call cleanup", timeout=25)
            async def no_bot_audio():
                return not any(participant.identity.startswith(bot_id + ":") for participant in room.remote_participants.values())
            await wait_until(no_bot_audio, "SFU participant cleanup")
            (paths / "rtc-peer-result.json").write_text(json.dumps({"mode": mode, "joined": joined,
                                                                   "left": {}, "received": decoded}))
        return {"joined": True, "left": True, "mode": mode}
    finally:
        for task in list(tasks):
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await room.disconnect()
        await matrix.close()
