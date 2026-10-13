# Recovered recipe; image reused, NOT rebuilt during the resumed proof.
FROM python:3.11
RUN python -m venv /opt/test-venv && /opt/test-venv/bin/pip install --no-cache-dir pytest==8.3.5 pytest-asyncio==0.25.3 pytest-timeout==2.3.1 pyyaml==6.0.3 python-dotenv==1.2.2 openai==2.24.0 rich==14.3.3 tenacity==9.1.4 httpx[socks]==0.28.1 jinja2==3.1.6 prompt-toolkit==3.0.52 croniter==6.0.0 psutil==7.2.2 fire==0.7.1 requests==2.33.0 websockets==15.0.1 pathspec==1.1.1 fastapi==0.135.1 ruamel.yaml==0.18.17 snowballstemmer==3.1.1
ENV HERMES_PYTHON=/opt/test-venv/bin/python HOME=/tmp/home
WORKDIR /repo
