# Apple Ecosystem Skills

Apple platform integration — Notes, Reminders, Messages, and Find My device tracking for macOS.

## Overview

This category contains 4 skills for integrating with Apple's ecosystem on macOS. These skills enable automation and programmatic access to Apple Notes, Reminders, iMessage, and Find My device tracking. Perfect for macOS users who want to automate their Apple workflows, build notification systems, or integrate Apple services into their development environment.

**Platform requirement:** macOS (these skills use native Apple apps and APIs)

## Available Skills

### Note Taking & Documentation

#### **apple-notes**
Manage Apple Notes via memo CLI — create, search, and edit notes programmatically.

**Use when:** Automating note-taking, capturing information to Notes, or building note-based workflows on macOS.

**Key features:**
- Create notes with memo CLI
- Search notes by content or title
- Edit existing notes
- Organize into folders
- Command-line automation
- Integration with Apple Notes.app

**Use cases:**
- Capture meeting notes automatically
- Save code snippets to Notes
- Create daily journal entries
- Search notes from terminal
- Backup important information

---

### Task Management

#### **apple-reminders**
Manage Apple Reminders via remindctl — add, list, and complete tasks.

**Use when:** Automating task management, creating reminders from scripts, or integrating with GTD workflows.

**Key features:**
- Add reminders with due dates
- List reminders by list or status
- Mark reminders as complete
- Set priorities and locations
- Command-line task management
- Sync across Apple devices via iCloud

**Use cases:**
- Create reminders from calendar events
- Set location-based reminders
- Build custom task workflows
- Automate recurring tasks
- Integrate with productivity systems

---

### Messaging & Communication

#### **imessage**
Send and receive iMessages and SMS via imsg CLI on macOS.

**Use when:** Automating messaging workflows, building notification systems, or sending iMessages from scripts.

**Key features:**
- Send iMessages to contacts or phone numbers
- Send SMS messages
- Read recent messages
- Group messaging support
- Attachment sending (images, files)
- Message history access
- CLI-based automation

**Use cases:**
- Alert notifications via iMessage
- Automated status updates
- Bot responses and automations
- Emergency notifications
- Team communication workflows
- Cross-device messaging

---

### Device Tracking

#### **findmy**
Track Apple devices and AirTags via FindMy.app on macOS.

**Use when:** Locating Apple devices, tracking AirTags, or building location-based automations.

**Key features:**
- Locate Apple devices (iPhone, iPad, Mac, Watch, AirPods)
- Track AirTag locations
- View device battery levels
- Play sound on devices
- Enable Lost Mode
- Get location coordinates
- Device status monitoring

**Use cases:**
- Device inventory management
- Lost device location tracking
- AirTag monitoring systems
- Location-based automations
- Device status dashboards
- Family device tracking

---

## Quick Start

### Example: Automated Note-Taking

```bash
# 1. Capture meeting notes
/apple-notes "Create note: 'Team Standup $(date)' with agenda and action items"

# 2. Search previous notes
/apple-notes "Search notes for 'project timeline'"

# 3. Update existing note
/apple-notes "Append to 'Project Tasks': New task - Review PR #123"
```

### Example: Task Automation

```bash
# 1. Create reminder from calendar event
/apple-reminders "Add reminder: Review quarterly report, due Friday 5pm"

# 2. List today's tasks
/apple-reminders "Show all reminders due today"

# 3. Complete tasks
/apple-reminders "Mark 'Review quarterly report' as complete"
```

### Example: Messaging Workflow

```bash
# 1. Send deployment notification
/imessage "Send to dev-team: Deployment to production complete ✅"

# 2. Emergency alert
/imessage "Send to on-call: Server down - investigating"

# 3. Daily standup reminder
/imessage "Send to team: Daily standup in 10 minutes"
```

### Example: Device Management

```bash
# 1. Locate devices
/findmy "Show location of all my devices"

# 2. Track specific device
/findmy "Where is my iPhone?"

# 3. Find AirTag
/findmy "Locate AirTag: Keys"
```

## Skill Combinations

**Complete Productivity System:**
1. Use `apple-notes` for documentation and ideas
2. Use `apple-reminders` for task management
3. Use `imessage` for team communication
4. Use `findmy` for device tracking

**Meeting Workflow:**
1. `imessage` — Send meeting reminder
2. `apple-notes` — Capture meeting notes
3. `apple-reminders` — Create action items
4. `imessage` — Share meeting summary

**Development Notifications:**
1. `imessage` — CI/CD status updates
2. `apple-notes` — Log deployment notes
3. `apple-reminders` — Create follow-up tasks

**Personal Automation:**
1. `apple-reminders` — Daily task review
2. `apple-notes` — Journal entry
3. `imessage` — Check-in messages
4. `findmy` — Device location check

## Choosing the Right Tool

**For capturing information:**
- Quick notes → `apple-notes`
- Tasks with due dates → `apple-reminders`
- Communication → `imessage`

**For communication:**
- One-way notifications → `imessage`
- Two-way conversations → `imessage` + read messages
- Team updates → `imessage` group messages

**For organization:**
- Documentation → `apple-notes`
- Action items → `apple-reminders`
- Device management → `findmy`

## Common Workflows

### Daily Productivity Routine

```bash
# Morning: Review tasks
/apple-reminders "List all tasks for today"

# Capture ideas throughout day
/apple-notes "Create note: Ideas - $(date)"

# Evening: Plan tomorrow
/apple-reminders "Add reminder: Review PRs, due tomorrow 10am"

# Device check before sleep
/findmy "Show battery status for all devices"
```

### Development Workflow

```bash
# Start work
/apple-notes "Create note: Dev Log - $(date)"

# Track tasks
/apple-reminders "Add: Implement auth feature, due today"

# During development
/apple-notes "Append to Dev Log: Fixed bug in user service"

# On deployment
/imessage "Send to team: v2.0 deployed to staging"

# End of day
/apple-notes "Append to Dev Log: Summary of progress"
/apple-reminders "Mark 'Implement auth feature' as complete"
```

### Team Communication

```bash
# Morning standup
/imessage "Send to dev-team: Standup starting now!"

# Share blockers
/imessage "Send to @manager: Blocked on API access for payment integration"

# Deployment notification
/imessage "Send to stakeholders: New feature live in production"

# Document in Notes
/apple-notes "Create note: Deployment - v2.1 - $(date)"
```

### Device Management

```bash
# Weekly device check
/findmy "List all devices with battery < 20%"

# Before travel
/findmy "Verify all devices are with me"

# Lost AirPods
/findmy "Play sound on AirPods Pro"
/findmy "Show last known location of AirPods Pro"

# Family devices
/findmy "Show location of family members' devices"
```

## Best Practices

**Apple Notes:**
- Use consistent naming conventions
- Organize with folders
- Tag important notes
- Regular backup/export
- Link related notes

**Apple Reminders:**
- Set realistic due dates
- Use priorities effectively
- Review and clean up regularly
- Use lists for categories
- Location-based when relevant

**iMessage:**
- Verify recipient before sending
- Keep messages concise
- Use group chats appropriately
- Respect messaging hours
- Check delivery status

**Find My:**
- Keep device names updated
- Enable Lost Mode when needed
- Check battery regularly
- Update device locations
- Use responsibly and ethically

## Platform Requirements

### macOS Version
- macOS 10.14+ recommended
- Newer versions for latest features
- System Integrity Protection considerations

### CLI Tools
- **memo** for Apple Notes
- **remindctl** for Reminders
- **imsg** for iMessage
- Find My accessed via Apple Events/scripting

### Permissions
Grant terminal access to:
- Apple Events
- Contacts (for iMessage)
- Location Services (for Find My)
- Full Disk Access (for some operations)

### iCloud
- iCloud account required for sync
- Two-factor authentication recommended
- Adequate iCloud storage

## Security & Privacy

**Data Access:**
- Skills access personal data (notes, messages, locations)
- Review permissions carefully
- Use encrypted storage for sensitive data
- Regular security audits

**Messaging:**
- Messages can contain sensitive information
- Verify recipients before automation
- Avoid sending passwords/secrets
- Use end-to-end encrypted channels

**Location Data:**
- Find My reveals device locations
- Use responsibly and ethically
- Respect privacy of family members
- Secure API access credentials

**Automation Security:**
- Review automated message content
- Test workflows before production
- Implement error handling
- Log automation actions

## Troubleshooting

### Common Issues

**Permissions Denied:**
```bash
# Grant terminal full disk access
System Settings → Privacy & Security → Full Disk Access → Add Terminal
```

**iMessage Not Sending:**
```bash
# Verify iMessage is enabled
Messages.app → Settings → iMessage → Enable
```

**Find My Not Working:**
```bash
# Enable Find My
System Settings → [Your Name] → Find My → Enable Find My Mac
```

**CLI Tools Not Found:**
```bash
# Install via Homebrew or package manager
brew install memo-cli
brew install remindctl
```

## Automation Examples

### Daily Journal

```bash
#!/bin/bash
# Create daily journal entry
DATE=$(date +"%Y-%m-%d")
/apple-notes "Create note: 'Journal - $DATE'"
/apple-notes "Append to 'Journal - $DATE': What I accomplished today..."
```

### Deployment Notification

```bash
#!/bin/bash
# Notify on successful deployment
if [ $? -eq 0 ]; then
  /imessage "Send to team: ✅ Deployment successful"
  /apple-notes "Create note: 'Deployment $VERSION - $DATE'"
else
  /imessage "Send to on-call: ❌ Deployment failed"
fi
```

### Device Battery Alert

```bash
#!/bin/bash
# Check device batteries
LOW_BATTERY=$(/findmy "List devices with battery < 15%")
if [ ! -z "$LOW_BATTERY" ]; then
  /imessage "Send to me: 🔋 Low battery alert: $LOW_BATTERY"
fi
```

## Contributing

Found a bug or have an enhancement idea?

1. Open an issue describing the improvement
2. Fork the repository
3. Make changes to the relevant `SKILL.md`
4. Submit a pull request

## Related Categories

- **productivity/** - Cross-platform productivity tools
- **note-taking/** - Advanced note-taking systems
- **github/** - Development workflows
- **software-development/** - Coding and debugging

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
