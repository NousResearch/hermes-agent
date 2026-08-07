# WhatsApp Setup Wizard (Español)

> Guía interactiva para configurar WhatsApp en Hermes Agent en 5 pasos.

`hermes whatsapp setup` es un asistente que reemplaza **4+ pasos manuales** por un solo comando. Configura el emparejamiento, restricciones de acceso, canal de entrega de cron y prueba la entrega de mensajes.

## Requisitos previos

- Hermes Agent instalado y configurado
- Node.js instalado (para el bridge de WhatsApp)
- WhatsApp en el celular

## Cómo usar

```bash
hermes whatsapp setup
```

El asistente lo guiará por 5 pasos:

---

## Paso 1/5 — Emparejamiento (Código QR)

El asistente inicia el bridge de WhatsApp y muestra un código QR.

1. Abra WhatsApp en su celular
2. Vaya a **Configuración → Dispositivos vinculados → Vincular un dispositivo**
3. Apunte la cámara al código QR mostrado en la terminal

**Si ya existe una sesión activa:** el asistente la detecta y pregunta si desea emparejar nuevamente. Responder "sí" limpia la sesión existente y genera un nuevo QR.

Después del emparejamiento, el número del bot se detecta automáticamente.

---

## Paso 2/5 — ¿Quién puede usar el bot?

Aquí define quién puede enviar mensajes al bot.

El asistente pregunta:

> "¿Este es tu número personal?"

- **Sí** — su número (detectado del emparejamiento) se agrega automáticamente a la lista de permitidos
- **No** — ingresa manualmente el número del dueño

Lo que se configura:
- `WHATSAPP_ALLOWED_USERS` → su número
- `WHATSAPP_ALLOW_ALL_USERS` → `false` (nadie más puede usar)
- `dm_policy` → `allowlist` (solo contactos autorizados)

> 💡 **Consejo:** El error más común es poner el número del bot en lugar del número del dueño en `WHATSAPP_ALLOWED_USERS`. El asistente lo hace automáticamente.

---

## Paso 3/5 — Canal de entrega (Cron)

Configure dónde se entregarán las tareas programadas (cron) en WhatsApp.

El asistente intenta **detectar automáticamente** su chat LID desde los logs del gateway. Si ya ha enviado un mensaje al bot, encuentra el LID correcto.

Si no puede detectarlo, ingresa el LID manualmente (ej: `55310773391517@lid`).

---

## Paso 4/5 — Prueba de entrega

El asistente envía un mensaje de prueba a través de la API HTTP del bridge (puerto 3000) para confirmar que todo funciona.

- ✅ **Éxito:** recibe un mensaje en WhatsApp
- ⚠ **Fallo:** el bridge puede no estar ejecutándose — inicie el gateway con `hermes gateway run` e intente de nuevo

---

## Paso 5/5 — Reiniciar gateway

Para que los cambios de configuración surtan efecto, el gateway debe reiniciarse.

El asistente pregunta si desea reiniciar ahora:

- **Sí** — ejecuta `systemctl restart hermes-gateway.service`
- **No** — debe reiniciar manualmente después:
  ```bash
  sudo systemctl restart hermes-gateway.service
  ```

---

## Resumen

Al final, el asistente muestra un resumen de todo lo configurado:

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

## Próximos pasos

1. Envíe un mensaje al bot en WhatsApp — responderá automáticamente
2. Programe tareas con cron:
   ```bash
   hermes cron create --schedule "every 1h" --prompt "Resumen del mercado hoy" --deliver whatsapp
   ```

## Comandos relacionados

| Comando | Descripción |
|---------|-------------|
| `hermes whatsapp` | Emparejamiento básico (sin asistente) |
| `hermes gateway run` | Iniciar el gateway |
| `hermes config set whatsapp.dm_policy allowlist` | Restringir acceso manualmente |
| `hermes cron create ... deliver whatsapp` | Programar tarea con entrega a WhatsApp |

## Solución de problemas

| Problema | Causa probable | Solución |
|----------|----------------|----------|
| El código QR no aparece | Node.js ausente | Instale Node.js 18+ |
| El mensaje de prueba falló | El bridge no está ejecutándose | Ejecute `hermes gateway run` |
| El bot no responde | El gateway no fue reiniciado | Reinicie con `sudo systemctl restart hermes-gateway.service` |
| "Número no autorizado" | dm_policy muy restrictivo | Reejecute `hermes whatsapp setup` y verifique el número del dueño |
