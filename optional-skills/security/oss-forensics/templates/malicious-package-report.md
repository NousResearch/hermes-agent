# Malicious Package Investigation Report

---

## 📦 Package Metadata
- **Package Name**: 
- **Registry**: [NPM / PyPI / RubyGems / etc.]
- **Affected Versions**: 
- **Malicious Version(s)**: 
- **Downloads at Time of Detection**: 
- **Package URL**: 

---

## 🚩 Indicators of Compromise (IOCs)
- **Malicious URL(s)**: 
- **Exfiltrated Data Types**: [Environment variables, ~/.ssh/id_rsa, /etc/shadow, etc.]
- **Exfiltration Method**: [DNS tunneling, HTTP POST to C2, etc.]
- **C2 IP/Domain**: 

---

## 🛠️ Analysis Summary
- **Primary Mechanism**: [Typosquatting / Dependency Confusion / Maintainer Takeover]
- **Behavior Description**: 
  - [Example: Installs a postinstall script that exfiltrates environment variables.]
  - [Example: Patches `setup.py` to download a secondary payload.]

---

## 🔍 Evidence Registry
| Evidence ID | Type | Source | Description |
|-------------|------|--------|-------------|
| EV-XXXX     | ioc  | NPM    | Package install script snapshot |
| EV-YYYY     | web  | Wayback| Historical version comparison |

---

## 🛡️ Recommended Mitigations
1. [ ] Unpublish/Report the package to the registry.
2. [ ] Audit `package-lock.json` or `requirements.txt` across all projects.
3. [ ] Rotate secrets exfiltrated via environment variables.
4. [ ] Pin specific hashes (SHASUM) for mission-critical dependencies.
