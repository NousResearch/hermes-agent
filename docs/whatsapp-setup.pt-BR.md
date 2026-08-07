# WhatsApp Setup Wizard (Português-BR)

> Guia interativo para configurar o WhatsApp no Hermes Agent em 5 passos.

`hermes whatsapp setup` é um wizard que substitui **4+ passos manuais** por um único comando. Ele configura o par de dispositivos, restrições de acesso, canal de entrega de cron e testa a entrega de mensagens.

## Pré-requisitos

- Hermes Agent instalado e configurado
- Node.js instalado (para a bridge do WhatsApp)
- WhatsApp no celular

## Como usar

```bash
hermes whatsapp setup
```

O wizard vai guiar você por 5 etapas:

---

## Passo 1/5 — Pareamento (QR Code)

O wizard inicia a bridge do WhatsApp e exibe um QR code.

1. Abra o WhatsApp no seu celular
2. Vá em **Configurações → Dispositivos Conectados → Conectar um Dispositivo**
3. Aponte a câmera para o QR code exibido no terminal

**Se já existe uma sessão ativa:** o wizard detecta e pergunta se você quer parear novamente. Responder "sim" limpa a sessão existente e gera um novo QR.

Após o pareamento, o número do bot é detectado automaticamente.

---

## Passo 2/5 — Quem pode usar o bot?

Aqui você define quem pode enviar mensagens para o bot.

O wizard pergunta:

> "Este é seu número pessoal?"

- **Sim** — seu número (detectado do pareamento) é adicionado à allowlist automaticamente
- **Não** — você digita manualmente o número do dono

O que é configurado:
- `WHATSAPP_ALLOWED_USERS` → seu número
- `WHATSAPP_ALLOW_ALL_USERS` → `false` (ninguém mais pode usar)
- `dm_policy` → `allowlist` (apenas contatos autorizados)

> 💡 **Dica:** O erro mais comum é colocar o número do bot em vez do número do dono no `WHATSAPP_ALLOWED_USERS`. O wizard faz isso automaticamente para você.

---

## Passo 3/5 — Canal de entrega (Cron)

Configure para onde as entregas de tarefas agendadas (cron) serão enviadas no WhatsApp.

O wizard tenta **detectar automaticamente** seu chat LID a partir dos logs do gateway. Se você já enviou uma mensagem para o bot, ele encontra o LID certo.

Se não conseguir detectar, você digita manualmente o LID (ex: `55310773391517@lid`).

---

## Passo 4/5 — Teste de entrega

O wizard envia uma mensagem de teste via API HTTP da bridge (porta 3000) para confirmar que tudo está funcionando.

- ✅ **Sucesso:** você recebe uma mensagem no WhatsApp
- ⚠ **Falha:** a bridge pode não estar rodando — inicie o gateway com `hermes gateway run` e tente novamente

---

## Passo 5/5 — Reiniciar gateway

Para as alterações de configuração entrarem em vigor, o gateway precisa ser reiniciado.

O wizard pergunta se você quer reiniciar agora:

- **Sim** — executa `systemctl restart hermes-gateway.service`
- **Não** — você precisa reiniciar manualmente depois:
  ```bash
  sudo systemctl restart hermes-gateway.service
  ```

---

## Resumo

Ao final, o wizard exibe um resumo de tudo que foi configurado:

```
✅ WhatsApp Setup Complete!

  ┌────────────────────────────────────────────────────────┐
  │  ✓ Paired as 5511999999999                             │
  │    Allowed users: 5511999999999 (only you)            │
  │    DM policy: allowlist (no randoms)                  │
  │    Home channel: 55310773391517@lid                   │
  │    Cron deliveries: enabled                           │
  │    Test delivery: ✅ PASS                              │
  │    Gateway: restart required                          │
  └────────────────────────────────────────────────────────┘
```

## Próximos passos

1. Envie uma mensagem para o bot no WhatsApp — ele responde automaticamente
2. Agende tarefas com cron:
   ```bash
   hermes cron create --schedule "every 1h" --prompt "Resumo do mercado hoje" --deliver whatsapp
   ```

## Comandos relacionados

| Comando | Descrição |
|---------|-----------|
| `hermes whatsapp` | Pareamento básico (sem wizard) |
| `hermes gateway run` | Iniciar o gateway |
| `hermes config set whatsapp.dm_policy allowlist` | Restringir acesso manualmente |
| `hermes cron create ... deliver whatsapp` | Agendar tarefa com entrega no WhatsApp |

## Solução de problemas

| Problema | Causa provável | Solução |
|----------|---------------|---------|
| QR code não aparece | Node.js ausente | Instale Node.js 18+ |
| Mensagem de teste falhou | Bridge não está rodando | Execute `hermes gateway run` |
| Bot não responde | Gateway não foi reiniciado | Reinicie com `sudo systemctl restart hermes-gateway.service` |
| "Número não autorizado" | dm_policy muito restritivo | Reexecute `hermes whatsapp setup` e verifique o número do dono |
