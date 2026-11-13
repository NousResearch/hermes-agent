# Verda Inference

Connect Hermes to Verda's OpenAI-compatible inference API.

## Setup

1. Enable a model in your Verda project and create an inference API key:
   https://docs.verda.com/inference/quickstart/
2. Run `hermes model` and select **Verda Inference**.
3. Enter your inference API key.
4. Keep the default base URL: `https://inference.verda.com/v1`.
5. Select a model available to your project.

For environment-based authentication, set `VERDA_INFERENCE_API_KEY`.

The provider discovers models through `/models`, with
`kimi-k3-1-pro` as a fallback when discovery is unavailable.
Auxiliary tasks also use `kimi-k3-1-pro` and incur normal inference charges.

## Reasoning

For `kimi-k3-1-pro`, the integration sends `low`, `high`, or `max`
as a top-level `reasoning_effort` parameter.

Other Hermes effort levels use its standard clamp:
`minimal` and `medium` map to `low`, `xhigh` maps to `high`,
and `ultra` maps to `max`.

Unset reasoning, disabled reasoning, and `none` omit the parameter,
leaving the server default in effect. This integration does not
currently implement a reasoning-off switch.

API documentation:
https://docs.verda.com/inference/openai-compatibility/