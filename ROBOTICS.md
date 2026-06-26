# Robotics applications for Hermes Agent IoT

This fork is well suited to robot systems as a lightweight edge agent, not as a hard real-time motor controller. Use it to coordinate high-level tasks, communicate with sensors/controllers, summarize state, run schedules/watchdogs, and bridge a remote or LAN AI model to the robot body.

## Recommended role

```text
Cloud/LAN LLM or operator UI = high-level brain
Hermes Agent IoT on Pi2/Pi3/Pi4 = edge coordinator / nervous system
MCU or ROS/ROS2 controller = reflexes and real-time motor/safety control
Sensors and actuators = robot body
```

Hermes Agent IoT should handle:

- natural-language task intake and high-level task decomposition
- MQTT/HTTP/serial/ROS-bridge communication
- sensor/status summarization for the LLM or operator
- task result reporting and event logging
- cron-based inspection, heartbeat, battery, and watchdog jobs
- safe command shaping before messages reach the robot controller

It should not directly expose raw motor PWM, unrestricted GPIO, or unbounded movement commands to an LLM.

## Practical architecture

```text
User / Web UI / remote LLM
        |
        v
Hermes Agent IoT on Pi-class Linux device
        |
        | MQTT / HTTP / serial
        v
Robot bridge or MCU controller
        |
        | GPIO / CAN / UART / I2C / SPI / ROS topics
        v
Motor drivers, sensors, gripper, lights, safety hardware
```

For Raspberry Pi 2-class hardware, keep large AI work remote:

- run LLM inference on a cloud provider, LAN workstation, mini PC, NAS, Pi 4/5, or other stronger node
- keep local Pi device duties lightweight: tools, MQTT, summaries, task state, watchdogs
- avoid local `torch`, `sentence-transformers`, `chromadb`, and large local models by default

## MQTT topic layout

The built-in MQTT toolset is a good first robot bus because it is lightweight and hardware-neutral.

Suggested topics:

```text
robot/base/cmd
robot/base/state
robot/arm/cmd
robot/arm/state
robot/gripper/cmd
robot/gripper/state
robot/sensors/battery
robot/sensors/temperature
robot/sensors/distance/front
robot/events
robot/heartbeat
robot/safety/estop
```

Example base command:

```json
{
  "action": "move",
  "linear_mps": 0.08,
  "angular_rps": 0.0,
  "duration_sec": 2.0,
  "require_clear_front_cm": 30
}
```

Example stop command:

```json
{"action":"stop"}
```

Example status message:

```json
{
  "battery_percent": 72,
  "front_distance_cm": 38,
  "left_motor": "ok",
  "right_motor": "ok",
  "temperature_c": 41,
  "safety_state": "clear",
  "last_command": "completed"
}
```

## Using the MQTT tools

Configure broker access:

```bash
export MQTT_HOST=192.168.1.10
export MQTT_PORT=1883
# Optional:
export MQTT_USERNAME=iot-user
export MQTT_PASSWORD=secret
export MQTT_TLS=false
hermes tools enable mqtt
```

Useful tool patterns:

- `mqtt_publish`: publish telemetry or send a command to `robot/.../cmd`
- `mqtt_subscribe_recent`: read retained or newly published status topics such as `robot/sensors/#`
- `mqtt_device_command`: publish a command and wait briefly for `robot/.../state` or an ack topic

Remember: MQTT brokers do not provide message history by default. Use retained messages for latest state, or add a separate persistence service if you need historical logs.

## Safety model

Do not let an LLM emit arbitrary low-level motor commands. Restrict it to safe high-level actions and validate again at the controller.

Recommended allowlist:

```text
stop
read_status
read_battery
read_distance
move_forward_slow
turn_left_slow
turn_right_slow
dock
set_led
open_gripper_limited
close_gripper_limited
```

Recommended limits:

```yaml
robot:
  max_linear_mps: 0.15
  max_angular_rps: 0.5
  max_move_duration_sec: 5
  require_front_clear_cm: 25
  require_ack: true
  allowed_actions:
    - stop
    - read_status
    - move
    - turn
    - dock
    - set_led
    - gripper
```

The low-level MCU or ROS controller should independently enforce:

- emergency stop state
- speed limits
- duration limits
- obstacle/bumper limits
- battery limits
- actuator range limits
- command timeout / dead-man switch

## ROS / ROS2 bridge pattern

Avoid making ROS2 a required Pi2 dependency. Instead, bridge Hermes/MQTT to ROS on a stronger robot controller when needed:

```text
Hermes Agent IoT
   |
   | MQTT / HTTP
   v
robot-bridge service
   |
   v
ROS/ROS2 topics and services
```

Example mappings:

```text
MQTT robot/base/cmd        -> ROS2 /cmd_vel
MQTT robot/sensors/battery <- ROS2 /battery_state
MQTT robot/events          <- ROS2 diagnostics/lifecycle events
```

This keeps the IoT profile portable across Arduino/ESP32/STM32 projects, simple mobile robots, and larger ROS robots.

## Watchdog and scheduled robot tasks

Hermes cron jobs can run lightweight checks:

- read `robot/heartbeat`
- read `robot/sensors/battery`
- check `robot/safety/estop`
- publish `robot/base/cmd {"action":"stop"}` if heartbeat is stale
- ask the robot to dock when battery is low
- summarize recent `robot/events` for an operator

Example policy:

```text
Every 30 seconds:
- if heartbeat is older than 60 seconds, send stop and alert
- if battery < 20%, send dock command
- if estop is active, do not send movement commands
```

## Suitable robot applications

Good fits:

- small indoor mobile robots
- lab or warehouse inspection robots
- agriculture/greenhouse monitoring robots
- home automation robots
- pan-tilt camera or sensor stations
- gripper/arm demos with strict movement limits
- educational robots using ESP32/Arduino/STM32 controllers
- robots that need remote AI planning but local lightweight control

Poor fits for Pi2-only local execution:

- direct high-rate motor control loops
- visual SLAM or neural perception on Pi2
- local 7B+ LLM inference on Pi2
- unrestricted autonomous navigation without a dedicated safety controller

## Suggested next implementation steps

If this fork grows more robot-specific functionality, implement it in layers:

1. Robot command schema
   - standard JSON for `move`, `stop`, `dock`, `read_status`, `set_led`, `gripper`
   - required safety fields such as max duration and ack requirement

2. Robot helper tools on top of MQTT
   - `robot_get_status`
   - `robot_send_command`
   - `robot_emergency_stop`
   - `robot_wait_for_ack`

3. Safety policy file
   - YAML limits for speed, duration, obstacle clearance, allowed actions, and ack requirements

4. Example bridge directory
   - `examples/robot_mqtt_bridge/`
   - Python simulator or ESP32/Arduino topic example
   - optional ROS2 bridge example

5. Watchdog/heartbeat examples
   - cron examples for heartbeat, battery, estop, and docking checks

The design goal is a clear split: Hermes Agent IoT coordinates and reasons; the robot controller enforces real-time safety.