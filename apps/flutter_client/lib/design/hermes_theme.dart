import 'package:flutter/material.dart';

class HermesColors extends ThemeExtension<HermesColors> {
  const HermesColors({
    required this.chrome,
    required this.sidebar,
    required this.editor,
    required this.elevated,
    required this.card,
    required this.input,
    required this.primaryText,
    required this.secondaryText,
    required this.tertiaryText,
    required this.quaternaryText,
    required this.accent,
    required this.accentSoft,
    required this.warm,
    required this.borderPrimary,
    required this.borderSecondary,
    required this.rowHover,
    required this.rowActive,
    required this.success,
    required this.warning,
    required this.danger,
  });

  final Color chrome;
  final Color sidebar;
  final Color editor;
  final Color elevated;
  final Color card;
  final Color input;
  final Color primaryText;
  final Color secondaryText;
  final Color tertiaryText;
  final Color quaternaryText;
  final Color accent;
  final Color accentSoft;
  final Color warm;
  final Color borderPrimary;
  final Color borderSecondary;
  final Color rowHover;
  final Color rowActive;
  final Color success;
  final Color warning;
  final Color danger;

  static const light = HermesColors(
    chrome: Color(0xFFF8FAFF),
    sidebar: Color(0xFFF3F7FF),
    editor: Color(0xFFFCFCFC),
    elevated: Color(0xFFFFFFFF),
    card: Color(0xFFF7F9FE),
    input: Color(0xFFFCFCFC),
    primaryText: Color(0xEF17171A),
    secondaryText: Color(0xBD17171A),
    tertiaryText: Color(0x8A17171A),
    quaternaryText: Color(0x5C17171A),
    accent: Color(0xFF0053FD),
    accentSoft: Color(0xFFEAF1FF),
    warm: Color(0xFFCF806D),
    borderPrimary: Color(0x33506EA6),
    borderSecondary: Color(0x24506EA6),
    rowHover: Color(0x140053FD),
    rowActive: Color(0x210053FD),
    success: Color(0xFF1F8A65),
    warning: Color(0xFFC08532),
    danger: Color(0xFFCF2D56),
  );

  static const dark = HermesColors(
    chrome: Color(0xFF111318),
    sidebar: Color(0xFF0A0A0B),
    editor: Color(0xFF161618),
    elevated: Color(0xFF1C1D21),
    card: Color(0xFF1A1C22),
    input: Color(0xFF161618),
    primaryText: Color(0xF5F7F8FF),
    secondaryText: Color(0xBDF7F8FF),
    tertiaryText: Color(0x8AF7F8FF),
    quaternaryText: Color(0x5CF7F8FF),
    accent: Color(0xFF4F8CFF),
    accentSoft: Color(0x263E7BFF),
    warm: Color(0xFFFF8C42),
    borderPrimary: Color(0x42B9C8FF),
    borderSecondary: Color(0x2EB9C8FF),
    rowHover: Color(0x1F4F8CFF),
    rowActive: Color(0x334F8CFF),
    success: Color(0xFF55A583),
    warning: Color(0xFFE0A14A),
    danger: Color(0xFFE75E78),
  );

  @override
  ThemeExtension<HermesColors> copyWith({
    Color? chrome,
    Color? sidebar,
    Color? editor,
    Color? elevated,
    Color? card,
    Color? input,
    Color? primaryText,
    Color? secondaryText,
    Color? tertiaryText,
    Color? quaternaryText,
    Color? accent,
    Color? accentSoft,
    Color? warm,
    Color? borderPrimary,
    Color? borderSecondary,
    Color? rowHover,
    Color? rowActive,
    Color? success,
    Color? warning,
    Color? danger,
  }) {
    return HermesColors(
      chrome: chrome ?? this.chrome,
      sidebar: sidebar ?? this.sidebar,
      editor: editor ?? this.editor,
      elevated: elevated ?? this.elevated,
      card: card ?? this.card,
      input: input ?? this.input,
      primaryText: primaryText ?? this.primaryText,
      secondaryText: secondaryText ?? this.secondaryText,
      tertiaryText: tertiaryText ?? this.tertiaryText,
      quaternaryText: quaternaryText ?? this.quaternaryText,
      accent: accent ?? this.accent,
      accentSoft: accentSoft ?? this.accentSoft,
      warm: warm ?? this.warm,
      borderPrimary: borderPrimary ?? this.borderPrimary,
      borderSecondary: borderSecondary ?? this.borderSecondary,
      rowHover: rowHover ?? this.rowHover,
      rowActive: rowActive ?? this.rowActive,
      success: success ?? this.success,
      warning: warning ?? this.warning,
      danger: danger ?? this.danger,
    );
  }

  @override
  ThemeExtension<HermesColors> lerp(
    covariant ThemeExtension<HermesColors>? other,
    double t,
  ) {
    if (other is! HermesColors) return this;
    return HermesColors(
      chrome: Color.lerp(chrome, other.chrome, t)!,
      sidebar: Color.lerp(sidebar, other.sidebar, t)!,
      editor: Color.lerp(editor, other.editor, t)!,
      elevated: Color.lerp(elevated, other.elevated, t)!,
      card: Color.lerp(card, other.card, t)!,
      input: Color.lerp(input, other.input, t)!,
      primaryText: Color.lerp(primaryText, other.primaryText, t)!,
      secondaryText: Color.lerp(secondaryText, other.secondaryText, t)!,
      tertiaryText: Color.lerp(tertiaryText, other.tertiaryText, t)!,
      quaternaryText: Color.lerp(quaternaryText, other.quaternaryText, t)!,
      accent: Color.lerp(accent, other.accent, t)!,
      accentSoft: Color.lerp(accentSoft, other.accentSoft, t)!,
      warm: Color.lerp(warm, other.warm, t)!,
      borderPrimary: Color.lerp(borderPrimary, other.borderPrimary, t)!,
      borderSecondary: Color.lerp(borderSecondary, other.borderSecondary, t)!,
      rowHover: Color.lerp(rowHover, other.rowHover, t)!,
      rowActive: Color.lerp(rowActive, other.rowActive, t)!,
      success: Color.lerp(success, other.success, t)!,
      warning: Color.lerp(warning, other.warning, t)!,
      danger: Color.lerp(danger, other.danger, t)!,
    );
  }
}

extension HermesThemeContext on BuildContext {
  HermesColors get hermesColors => Theme.of(this).extension<HermesColors>()!;
}

ThemeData hermesTheme(Brightness brightness) {
  final isDark = brightness == Brightness.dark;
  final colors = isDark ? HermesColors.dark : HermesColors.light;
  final scheme = ColorScheme.fromSeed(
    seedColor: colors.accent,
    brightness: brightness,
    primary: colors.accent,
    secondary: colors.warm,
    surface: colors.editor,
    error: colors.danger,
  );
  final textTheme = _textTheme(colors);

  return ThemeData(
    useMaterial3: true,
    brightness: brightness,
    colorScheme: scheme,
    scaffoldBackgroundColor: colors.chrome,
    fontFamily: _sansStack,
    textTheme: textTheme,
    extensions: [colors],
    appBarTheme: AppBarTheme(
      elevation: 0,
      scrolledUnderElevation: 0,
      centerTitle: false,
      backgroundColor: colors.chrome,
      foregroundColor: colors.primaryText,
      titleTextStyle: textTheme.titleMedium,
      iconTheme: IconThemeData(color: colors.secondaryText, size: 19),
    ),
    cardTheme: CardThemeData(
      color: colors.editor,
      elevation: 0,
      margin: EdgeInsets.zero,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(10),
        side: BorderSide(color: colors.borderSecondary),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: colors.input,
      labelStyle: TextStyle(color: colors.tertiaryText, fontSize: 12),
      hintStyle: TextStyle(color: colors.quaternaryText, fontSize: 12),
      prefixIconColor: colors.tertiaryText,
      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 11),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(9),
        borderSide: BorderSide(color: colors.borderSecondary),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(9),
        borderSide: BorderSide(color: colors.borderSecondary),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(9),
        borderSide: BorderSide(color: colors.borderPrimary),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(9),
        borderSide: BorderSide(color: colors.danger),
      ),
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        backgroundColor: colors.accent,
        foregroundColor: Colors.white,
        minimumSize: const Size(0, 38),
        padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 10),
        textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(9)),
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        foregroundColor: colors.primaryText,
        side: BorderSide(color: colors.borderSecondary),
        minimumSize: const Size(0, 38),
        padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 10),
        textStyle: const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(9)),
      ),
    ),
    iconButtonTheme: IconButtonThemeData(
      style: IconButton.styleFrom(
        foregroundColor: colors.secondaryText,
        hoverColor: colors.rowHover,
        highlightColor: colors.rowActive,
      ),
    ),
    listTileTheme: ListTileThemeData(
      dense: true,
      iconColor: colors.tertiaryText,
      textColor: colors.primaryText,
      selectedColor: colors.primaryText,
      selectedTileColor: colors.rowActive,
      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(9)),
    ),
    dividerTheme: DividerThemeData(color: colors.borderSecondary, space: 1),
  );
}

TextTheme _textTheme(HermesColors colors) {
  final base = TextStyle(color: colors.primaryText, fontFamily: _sansStack);
  return TextTheme(
    displaySmall: base.copyWith(
        fontSize: 28, fontWeight: FontWeight.w700, letterSpacing: -0.6),
    headlineSmall: base.copyWith(
        fontSize: 21, fontWeight: FontWeight.w600, letterSpacing: -0.3),
    titleLarge: base.copyWith(
        fontSize: 17, fontWeight: FontWeight.w600, letterSpacing: -0.2),
    titleMedium: base.copyWith(fontSize: 14, fontWeight: FontWeight.w600),
    titleSmall: base.copyWith(fontSize: 12, fontWeight: FontWeight.w600),
    bodyLarge: base.copyWith(fontSize: 13, height: 1.5),
    bodyMedium: base.copyWith(fontSize: 12.5, height: 1.45),
    bodySmall:
        base.copyWith(fontSize: 11.5, height: 1.35, color: colors.tertiaryText),
    labelLarge: base.copyWith(fontSize: 12, fontWeight: FontWeight.w600),
    labelMedium: base.copyWith(
        fontSize: 11, fontWeight: FontWeight.w600, color: colors.secondaryText),
    labelSmall: base.copyWith(
        fontSize: 10, fontWeight: FontWeight.w600, color: colors.tertiaryText),
  );
}

const _sansStack = 'Segoe UI';
const hermesMonoStack = 'JetBrains Mono';
