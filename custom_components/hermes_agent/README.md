# 🏠 Hermes Agent - Home Assistant Integration

Integrazione di Hermes Agent (l'agente IA di Nous Research) in Home Assistant per automatizzazioni intelligenti e assistenza vocale avanzata.

## 📋 Requisiti

- **Home Assistant** 2024.12.0 o superiore
- **Hermes Agent** installato e in esecuzione
- Gateway Hermes esposto su una porta accessibile
- Chiave API per il provider LLM (OpenAI, Anthropic, ecc.)

## 🚀 Installazione

### 1. Setup Hermes Agent
```bash
# Installazione
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# Configurare il gateway
hermes setup
hermes gateway start
```

### 2. Copiare i file in Home Assistant
```bash
# Copia l'integrazione
cp -r custom_components/hermes_agent ~/.homeassistant/config/custom_components/

# Riavvia Home Assistant
```

### 3. Aggiungere l'integrazione
1. Vai a **Impostazioni → Dispositivi e servizi**
2. Clicca **"Crea integrazione"**
3. Cerca **"Hermes Agent"**
4. Inserisci:
   - **Gateway URL**: localhost (o IP del server)
   - **Gateway Port**: 8080 (porta del gateway)
   - **Provider**: openai, anthropic, ecc.
   - **Model**: gpt-4, claude-3-sonnet, ecc.
   - **API Key**: La tua chiave API

## 📊 Entità Disponibili

### Sensori
- **hermes_agent_status** - Stato del gateway (online/offline/busy/idle)
- **hermes_agent_session** - Informazioni sulla sessione corrente

## 🔧 Servizi Disponibili

### service.hermes_agent.chat
Invia un messaggio all'agente:
```yaml
service: hermes_agent.chat
data:
  message: "Qual è il meteo?"
```

### service.hermes_agent.execute_skill
Esegui una skill:
```yaml
service: hermes_agent.execute_skill
data:
  skill_name: "python_script"
  args:
    code: "print('Hello Home Assistant')"
```

### service.hermes_agent.query
Fai una domanda e ricevi una risposta:
```yaml
service: hermes_agent.query
data:
  message: "Dimmi 3 curiosità sul sistema solare"
```

## 📝 Esempi di Automazioni

### Accendi la luce con comando vocale
```yaml
automation:
  - alias: "Hermes - Accendi luce salotto"
    trigger:
      platform: template
      value_template: "{{ state_attr('sensor.hermes_agent_status', 'last_command') == 'light on' }}"
    action:
      - service: light.turn_on
        target:
          entity_id: light.salotto
```

### Riepilogo giornaliero con Hermes
```yaml
automation:
  - alias: "Hermes - Briefing mattutino"
    trigger:
      platform: time
      at: "07:00:00"
    action:
      - service: hermes_agent.query
        data:
          message: "Dammi un briefing delle notizie di oggi"
      - service: notify.telegram
        data:
          message: "{{ states('sensor.hermes_agent_session') }}"
```

## 🔐 Sicurezza

- Usa una **password forte** per Home Assistant
- Se accedi da remoto, usa **HTTPS** e **VPN**
- Mantieni la **API Key segura** - usa i segreti di Home Assistant
- Il gateway dovrebbe essere in una **rete privata**

## 🐛 Troubleshooting

### Errore: "Failed to connect to gateway"
```bash
# Verifica che il gateway è in esecuzione
hermes gateway start

# Verifica la porta
netstat -tlnp | grep 8080

# Testa la connessione
curl http://localhost:8080/health
```

### Errore: "Invalid API key"
- Verifica la chiave API nel setup
- Accedi a `~/.hermes/config.toml` e controlla le credenziali
- Riconfigura l'integrazione tramite Impostazioni

### Sensori offline
```bash
# Riavvia il gateway
hermes gateway stop
hermes gateway start

# Riavvia l'integrazione da HA
```

## 📚 Documentazione

- [Hermes Agent Docs](https://hermes-agent.nousresearch.com/docs/)
- [Home Assistant Integrations](https://www.home-assistant.io/integrations/)
- [Repository](https://github.com/thcuba/HAOS-hermes-agent)

## 💡 Prossimi Passi

- [ ] Aggiungere entità text per chat inline
- [ ] Supporto per la transcodifica audio (voice commands)
- [ ] Dashboard avanzato con statistiche
- [ ] Automazioni basate su skills
- [ ] Integrazione con scene e routine

## 📄 Licenza

MIT - Vedi [LICENSE](../../LICENSE)

---

**Creato da**: [thcuba](https://github.com/thcuba)  
**Based on**: [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research
