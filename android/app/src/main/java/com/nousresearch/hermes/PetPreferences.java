package com.nousresearch.hermes;

import android.content.Context;
import android.content.SharedPreferences;

/** Persistent preferences for the user-selected Hermes Pet mascot. */
public final class PetPreferences {
    private static final String PREFS = "hermes_pet_preferences";
    private static final String MASCOT = "mascot";

    private PetPreferences() {}

    public static String getMascot(Context context) {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getString(MASCOT, "🐈");
    }

    public static void setMascot(Context context, String mascot) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit().putString(MASCOT, mascot).apply();
    }

    public static boolean isEnabled(Context context) {
        return context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .getBoolean("enabled", false);
    }

    public static void setEnabled(Context context, boolean enabled) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit().putBoolean("enabled", enabled).apply();
    }
}
