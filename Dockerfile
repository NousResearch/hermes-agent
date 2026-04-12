FROM debian:13.4

# Disable Python stdout buffering to ensure logs are printed immediately
ENV PYTHONUNBUFFERED=1

ARG USERNAME=hermes
ARG UID=1000


# Install system dependencies in one layer, clear APT cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sudo bash curl git build-essential nodejs npm python3 python3-pip ripgrep ffmpeg gcc python3-dev libffi-dev procps && \
    rm -rf /var/lib/apt/lists/*

#Install ttyd
RUN ARCH=$(uname -m) && \
    curl -sL https://github.com/tsl0922/ttyd/releases/latest/download/ttyd.${ARCH} -o /usr/local/bin/ttyd && \
    chmod +x /usr/local/bin/ttyd

# --- Set up non-root user with given username ---
RUN groupadd -g ${UID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${USERNAME} -s /bin/bash ${USERNAME} && \
    echo "$${USERNAME} ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers

#Copy source to /opt/hermes
COPY . /${USERNAME}/hermes
WORKDIR /${USERNAME}/hermes

#Make /root bin folder
RUN mkdir -p /${USERNAME}/.local/bin
ENV PATH="/${USERNAME}/.local/bin:$PATH"

#Get uv prebuilt executable
COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

# Tell uv to put its downloaded Pythons in a place everyone can read
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
#Build files in /opt/hermes
RUN uv venv /${USERNAME}/hermes/.venv --python 3.11 && \
    . /${USERNAME}/hermes/.venv/bin/activate && \
    uv pip install -e ".[all]" && \
    npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    cd /${USERNAME}/hermes/scripts/whatsapp-bridge && \
    npm install --prefer-offline --no-audit && \
    npm cache clean --force
    #uv pip install -e "./tinker-atropos"

WORKDIR /${USERNAME}/hermes
#Not needed in this setup flow but leftover from main branch
RUN chmod +x /${USERNAME}/hermes/docker/entrypoint.sh

#Symbolic link hermes executable for client username
RUN ln -sf "/${USERNAME}/hermes/.venv/bin/hermes" /${USERNAME}/.local/bin/hermes
ENV PATH="/${USERNAME}/.local/bin:$PATH"

#Create workspace directories
RUN mkdir -p /${USERNAME}/data
ENV HERMES_HOME=/${USERNAME}/data
WORKDIR /${USERNAME}/data

EXPOSE 7681
# --- Configure ownership for client user---
RUN chown -R ${USERNAME}:${USERNAME} /${USERNAME}
RUN chmod -R 777 /${USERNAME}
USER ${USERNAME}
VOLUME [ "/${USERNAME}/data" ]
#CMD ["ttyd", "-W", "-p", "7681", "hermes gateway run"]