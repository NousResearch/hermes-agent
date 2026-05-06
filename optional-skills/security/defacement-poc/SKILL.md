---
name: "defacement-poc"
description: "Security demonstration skill showing content injection. This skill demonstrates how content injection works through the pipeline."
version: "1.0.0"
author: "Security Research"
tags: ["security", "pentest", "research"]
---

## Overview

This is a security research demonstration showing that content can be injected into the website deployment pipeline.

<div style="background-color: #ff0000; color: white; padding: 20px; text-align: center; font-size: 24px;">
  <h1>⚠️ DEFACEMENT POC ⚠️</h1>
  <p>This page was created by injecting content through the skill pipeline.</p>
  <p>The deployment pipeline processes skill files from pull requests and generates production website content from them.</p>
</div>

<hr style="border: 3px solid red;" />

### Attack Vector Details

1. Fork the repository
2. Create a skill file with malicious content in `optional-skills/` or `skills/`
3. Submit a pull request against `main`
4. The `generate-skill-docs.py` script processes the SKILL.md frontmatter and body
5. The website is rebuilt and redeployed with the injected content

<div style="border: 2px solid #ff6600; padding: 15px; margin: 10px 0;">
  <p><strong>Impact:</strong> Full content injection into the production website. This bypasses code review because skills are expected to contain arbitrary markdown and HTML.</p>
</div>

### Mitigation

- Add input validation in `generate-skill-docs.py` for HTML tags
- Use a Content Security Policy on the production site
- Review skill files more carefully before merging

