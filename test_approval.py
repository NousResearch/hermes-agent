import os
import threading
import time
from tools.approval import (
    check_all_command_guards, register_gateway_notify, 
    resolve_gateway_approval, unregister_gateway_notify,
    set_current_session_key, reset_current_session_key
)

session_key = "test-session"
notified = []
result_holder = [None]

def notify_callback(data):
    print(f"Notification received: {data}")
    notified.append(data)

# Register the notification callback
register_gateway_notify(session_key, notify_callback)

def agent_thread():
    from tools.approval import get_current_session_key, _gateway_notify_cbs, _lock
    print("Agent thread started")
    token = set_current_session_key(session_key)
    os.environ["HERMES_GATEWAY_SESSION"] = "1"
    os.environ["HERMES_EXEC_ASK"] = "1"
    os.environ["HERMES_SESSION_KEY"] = session_key
    try:
        print(f"Current session key: {get_current_session_key()}")
        with _lock:
            print(f"Registered callbacks: {_gateway_notify_cbs}")
        print("Calling check_all_command_guards")
        result = check_all_command_guards("rm -rf /important", "local")
        print(f"Check result: {result}")
        result_holder[0] = result
    finally:
        os.environ.pop("HERMES_GATEWAY_SESSION", None)
        os.environ.pop("HERMES_EXEC_ASK", None)
        os.environ.pop("HERMES_SESSION_KEY", None)
        reset_current_session_key(token)
        print("Agent thread finished")

# Start the agent thread
t = threading.Thread(target=agent_thread)
t.start()

# Wait for notification
print("Waiting for notification...")
for i in range(50):
    if notified:
        print(f"Notification received after {i} iterations")
        break
    time.sleep(0.05)
else:
    print("No notification received")

# Print results
print(f"Notified: {notified}")
print(f"Result holder: {result_holder}")

# Clean up
unregister_gateway_notify(session_key)
t.join(timeout=5)
