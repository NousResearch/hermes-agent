import 'package:flutter/material.dart';

import 'hermes_theme.dart';

class HermesShell extends StatelessWidget {
  const HermesShell({
    super.key,
    required this.title,
    required this.children,
    this.subtitle,
    this.actions = const [],
    this.leading,
  });

  final String title;
  final String? subtitle;
  final List<Widget> children;
  final Widget? leading;
  final List<Widget> actions;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              colors.chrome,
              Color.alphaBlend(colors.accentSoft, colors.chrome)
            ],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 920),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _ShellHeader(
                      title: title,
                      subtitle: subtitle,
                      leading: leading,
                      actions: actions,
                    ),
                    const SizedBox(height: 14),
                    Expanded(
                      child: ListView(
                        children: [
                          for (final child in children) ...[
                            child,
                            if (child != children.last)
                              const SizedBox(height: 12),
                          ],
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _ShellHeader extends StatelessWidget {
  const _ShellHeader({
    required this.title,
    this.subtitle,
    this.leading,
    this.actions = const [],
  });

  final String title;
  final String? subtitle;
  final Widget? leading;
  final List<Widget> actions;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    return Row(
      children: [
        leading ?? const HermesGlyph(),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: Theme.of(context).textTheme.titleLarge),
              if (subtitle != null) ...[
                const SizedBox(height: 2),
                Text(
                  subtitle!,
                  style: Theme.of(context)
                      .textTheme
                      .bodySmall
                      ?.copyWith(color: colors.tertiaryText),
                ),
              ],
            ],
          ),
        ),
        ...actions,
      ],
    );
  }
}

class HermesGlyph extends StatelessWidget {
  const HermesGlyph({super.key, this.size = 32});

  final double size;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(size * 0.28),
        border: Border.all(color: colors.borderSecondary),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [colors.accent, colors.warm],
        ),
        boxShadow: [
          BoxShadow(
            color: colors.accent.withAlpha(38),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Icon(Icons.auto_awesome, color: Colors.white, size: size * 0.52),
    );
  }
}

class HermesPanel extends StatelessWidget {
  const HermesPanel({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(14),
    this.selected = false,
  });

  final Widget child;
  final EdgeInsetsGeometry padding;
  final bool selected;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: selected
            ? Color.alphaBlend(colors.rowActive, colors.editor)
            : colors.editor,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
            color: selected ? colors.borderPrimary : colors.borderSecondary),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha(10),
            blurRadius: 18,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: child,
    );
  }
}

class HermesBadge extends StatelessWidget {
  const HermesBadge({
    super.key,
    required this.label,
    this.icon,
    this.tone = HermesBadgeTone.neutral,
  });

  final String label;
  final IconData? icon;
  final HermesBadgeTone tone;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    final toneColor = switch (tone) {
      HermesBadgeTone.neutral => colors.accent,
      HermesBadgeTone.success => colors.success,
      HermesBadgeTone.warning => colors.warning,
      HermesBadgeTone.danger => colors.danger,
      HermesBadgeTone.warm => colors.warm,
    };
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: toneColor.withAlpha(24),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: toneColor.withAlpha(58)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: 12, color: toneColor),
            const SizedBox(width: 4),
          ],
          Text(
            label,
            style: Theme.of(context)
                .textTheme
                .labelSmall
                ?.copyWith(color: toneColor),
          ),
        ],
      ),
    );
  }
}

enum HermesBadgeTone { neutral, success, warning, danger, warm }

class HermesCodeText extends StatelessWidget {
  const HermesCodeText(this.text, {super.key, this.color});

  final String text;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    final colors = context.hermesColors;
    return Text(
      text,
      overflow: TextOverflow.ellipsis,
      style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: color ?? colors.secondaryText,
            fontFamily: hermesMonoStack,
            fontSize: 11,
          ),
    );
  }
}
