from gateway import hosted_room_driver as driver
from tests.gateway.test_hosted_discussion_file_context import service as service_fixture


service = service_fixture

def test_send_without_a_thread_starts_its_own(service):
    event = service.send(room_id="room", event_id="user-plain", payload={"text": "@writer hello"})
    assert event["payload"]["thread_id"] == "user-plain"
    explicit = service.send(room_id="room", event_id="user-reply",
                            payload={"text": "@writer more", "thread_id": "user-plain"})
    assert explicit["payload"]["thread_id"] == "user-plain"
    task, = driver.list_tasks(service.db_path, room_id="room", status="queued")
    assert task["identity"].thread_id == "user-plain"
