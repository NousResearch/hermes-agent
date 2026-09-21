import json

p = '/home/kensei/.hermes/data/pending-investigation.json'
with open(p, 'r') as f:
    data = json.load(f)

report1 = """## Recommendation Report
### t_d6edfbdd — Ops Board Dispatcher Not Ticking Since 00:16
**What's actually happening:** The ops board dispatcher last ticked at 00:16 on 30/05 and was silent for ~14 hours. Research dispatcher kept running. At ~14:06 and 14:22 today the ops dispatcher resumed ticking and is now dispatching normally (spawned 4 workers at 14:06, 3 at 14:22). The thread likely auto-recovered after a long stall or a gateway event.
| # | Option | What happens |
| A | Close as transient — dispatcher healthy now | No further action; accept that intermittent stall recovered |
| B | Root-cause chase — grep logs around 00:16 tonight | Read 30/05 gateway.log for crash/restart signals before the stall, to see if we can prevent recurrence |
| C | Add dispatcher health watchdog | Run a cheap cron every 15 min checking last ops-dispatcher tick; alert after 1 hour silence |
**Default recommendation:** C — dispatcher lost 14 hours without anyone noticing; a watchdog turns invisible recovery into intentional visibility."""

report2 = """## Recommendation Report
### t_a470c23f — Provider Degradation: kimi-k2.6 HTTP 401 on ollama-cloud
**What's actually happening:** The ollama-cloud provider returned HTTP 401 for kimi-k2.6 at 09:12 on 30/05. This is not a rate limit — it's an invalid key, blocked account, or exhausted funds. Deepseek-v4-pro is also rate-limited (weekly cap) and qwen3.6-plus on 'nous' is out of credits. A 401 + 429 + 404 cascade means profiles routing through these providers will experience total failure for all dispatched workers until at least one provider recovers.
| # | Option | What happens |
| A | Verify / rotate Ollama Cloud API key | Log into https://ollama.com/settings, replace OLLAMA_API_KEY in ~/.hermes/.env, then dispatch a test worker |
| B | Switch auxiliary extract model to Gemini/NVIDIA | Change AUXILIARY_WEB_EXTRACT_MODEL to a provider with a working key (e.g., gemini-2-flash or NVIDIA variant) |
| C | Drop kimi-k2.6 from board profiles and set a known-good model as primary for ~20 affected profiles | Reprofile primary models to avoid the cascade entirely |
**Default recommendation:** A — start at the source; the 401 almost always means the Ollama Cloud key is expired / out of funds. If portal shows key is valid, then fall to B to unblock the pipeline."""

for task in data.get('tasks', []):
    if task['id'] == 't_d6edfbdd':
        task['investigated'] = True
        task['recommendations'] = report1
    if task['id'] == 't_a470c23f':
        task['investigated'] = True
        task['recommendations'] = report2

with open(p, 'w') as f:
    json.dump(data, f, indent=2)

print('updated')
