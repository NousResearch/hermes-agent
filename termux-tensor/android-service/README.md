# Android LiteRT-LM Service Contract V2

Termux remains the Hermes agent/orchestration layer. Native Android owns LiteRT-LM and Android-supported CPU/GPU/NPU delegates.

Endpoint: http://127.0.0.1:9380/v1

GET /health
GET /models
POST /chat/completions
POST /stop

Backend values: AUTO, CPU, GPU, NPU.
AUTO preference: NPU, then GPU, then CPU.
The service must report the actual backend selected and must not claim accelerator access without runtime confirmation.

Security: loopback only by default. LAN exposure is opt-in. Long-running background inference uses an Android foreground service.

Thermal modes: BALANCED, PERFORMANCE, COOL, BATTERY_SAVER.

Default: one loaded model at a time.

Recommended Android stack: Kotlin, LiteRT-LM Android API, Foreground Service, Binder or localhost HTTP bridge, Jetpack Compose management UI.
