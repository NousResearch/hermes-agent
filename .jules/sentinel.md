## 2025-05-07 - [Fix] Insecure XML Parsing in WeCom Gateway
**Vulnerability:** Found `xml.etree.ElementTree` being used to parse incoming external Webhooks in `gateway/platforms/wecom_callback.py`. This is vulnerable to XXE (XML External Entity) attacks and Billion Laughs.
**Learning:** External webhook handling parsing raw XML requires `defusedxml` to prevent entities expansion, especially since WeCom callback sends untrusted payloads from the internet directly to the agent runtime.
**Prevention:** Always use `defusedxml.ElementTree` for parsing XML, and specifically restrict parsing of untrusted inputs.
