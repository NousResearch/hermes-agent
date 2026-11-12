# Unitalk model provider

This bundled provider connects Hermes to the Unitalk gateway at
`https://llmgateway.unitalk.ai/v1`. Set `UNITALK_API_KEY` in the active
profile's `.env` or environment. If it is unset, `OPENAI_API_KEY` is used
instead; it must be a key accepted by the Unitalk gateway.

Run `hermes model`, select Unitalk, and choose a model from the gateway's live
catalog. The gateway controls which models and capabilities are available.

The standalone distribution is maintained at
https://github.com/unitalkai/unitalk-gateway-provider-plugin. This bundled copy
is updated with the Unitalk Desktop runtime, not by the plugin installer.
