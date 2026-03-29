FROM debian:13.4

RUN apt-get update
RUN apt-get install -y nodejs npm python3 python3-pip ripgrep ffmpeg gcc python3-dev libffi-dev

COPY . /opt/hermes
WORKDIR /opt/hermes

RUN pip install -e ".[all]" --break-system-packages
RUN npm install
RUN npx playwright install --with-deps chromium
WORKDIR /opt/hermes/scripts/whatsapp-bridge
RUN npm install

WORKDIR /opt/hermes
RUN chmod +x /opt/hermes/docker/entrypoint.sh

ENV HERMES_HOME=/opt/data
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/opt/hermes/docker/entrypoint.sh" ]