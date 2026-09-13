# High-value evaluation lanes

Hermes uses three independent evidence lanes: context-compression probes, core-tool A/B performance tests, and isolated Harbor/Gym research environments. These lanes produce reports and recommendations; they do not change runtime configuration.

Model-backed runs require explicit credentials, pinned source/model configuration, bounded cost, and redacted artifacts. Local smoke tests validate contracts without credentials. A local pass is not hosted-CI evidence.
