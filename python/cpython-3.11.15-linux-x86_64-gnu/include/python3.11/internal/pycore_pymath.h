#ifndef Py_INTERNAL_PYMATH_H
#define Py_INTERNAL_PYMATH_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


/* _Py_ADJUST_ERANGE1(x)
 * _Py_ADJUST_ERANGE2(x, y)
 * Set errno to 0 before calling a libm function, and invoke one of these
 * macros after, passing the function result(s) (_Py_ADJUST_ERANGE2 is useful
 * for functions returning complex results).  This makes two kinds of
 * adjustments to errno:  (A) If it looks like the platform libm set
 * errno=ERANGE due to underflow, clear errno. (B) If it looks like the
 * platform libm overflowed but didn't set errno, force errno to ERANGE.  In
 * effect, we're trying to force a useful implementation of C89 errno
 * behavior.
 * Caution:
 *    This isn't reliable.  C99 no longer requires libm to set errno under
 *        any exceptional condition, but does require +- HUGE_VAL return
 *        values on overflow.  A 754 box *probably* maps HUGE_VAL to a
 *        double infinity, and we're cool if that's so, unless the input
 *        was an infinity and an infinity is the expected result.  A C89
 *        system sets errno to ERANGE, so we check for that too.  We're
 *        out of luck if a C99 754 box doesn't map HUGE_VAL to +Inf, or
 *        if the returned result is a NaN, or if a C89 box returns HUGE_VAL
 *        in non-overflow cases.
 */
static inline void _Py_ADJUST_ERANGE1(double x)
{
    if (errno == 0) {
        if (x == Py_HUGE_VAL || x == -Py_HUGE_VAL) {
            errno = ERANGE;
        }
    }
    else if (errno == ERANGE && x == 0.0) {
        errno = 0;
    }
}

static inline void _Py_ADJUST_ERANGE2(double x, double y)
{
    if (x == Py_HUGE_VAL || x == -Py_HUGE_VAL ||
        y == Py_HUGE_VAL || y == -Py_HUGE_VAL)
    {
        if (errno == 0) {
            errno = ERANGE;
        }
    }
    else if (errno == ERANGE) {
        errno = 0;
    }
}

// Return whether integral type *type* is signed or not.
#define _Py_IntegralTypeSigned(type) \
    ((type)(-1) < 0)

// Return the maximum value of integral type *type*.
#define _Py_IntegralTypeMax(type) \
    ((_Py_IntegralTypeSigned(type)) ? (((((type)1 << (sizeof(type)*CHAR_BIT - 2)) - 1) << 1) + 1) : ~(type)0)

// Return the minimum value of integral type *type*.
#define _Py_IntegralTypeMin(type) \
    ((_Py_IntegralTypeSigned(type)) ? -_Py_IntegralTypeMax(type) - 1 : 0)

// Check whether *v* is in the range of integral type *type*. This is most
// useful if *v* is floating-point, since demoting a floating-point *v* to an
// integral type that cannot represent *v*'s integral part is undefined
// behavior.
#define _Py_InIntegralTypeRange(type, v) \
    (_Py_IntegralTypeMin(type) <= v && v <= _Py_IntegralTypeMax(type))


//--- HAVE_PY_SET_53BIT_PRECISION macro ------------------------------------
//
// The functions _Py_dg_strtod() and _Py_dg_dtoa() in Python/dtoa.c (which are
// required to support the short float repr introduced in Python 3.1) require
// that the floating-point unit that's being used for arithmetic operations on
// C doubles is set to use 53-bit precision.  It also requires that the FPU
// rounding mode is round-half-to-even, but that's less often an issue.
//
// If your FPU isn't already set to 53-bit precision/round-half-to-even, and
// you want to make use of _Py_dg_strtod() and _Py_dg_dtoa(), then you should:
//
//     #define HAVE_PY_SET_53BIT_PRECISION 1
//
// and also give appropriate definitions for the following three macros:
//
// * _Py_SET_53BIT_PRECISION_HEADER: any variable declarations needed to
//   use the two macros below.
// * _Py_SET_53BIT_PRECISION_START: store original FPU settings, and
//   set FPU to 53-bit precision/round-half-to-even
// * _Py_SET_53BIT_PRECISION_END: restore original FPU settings
//
// The macros are designed to be used within a single C function: see
// Python/pystrtod.c for an example of their use.


// Get and set x87 control word for gcc/x86
#ifdef HAVE_GCC_ASM_FOR_X87
#define HAVE_PY_SET_53BIT_PRECISION 1

// Functions defined in Python/pymath.c
extern unsigned short _Py_get_387controlword(void);
extern void _Py_set_387controlword(unsigned short);

#define _Py_SET_53BIT_PRECISION_HEADER                                  \
    unsigned short old_387controlword, new_387controlword
#define _Py_SET_53BIT_PRECISION_START                                   \
    do {                                                                \
        old_387controlword = _Py_get_387controlword();                  \
        new_387controlword = (old_387controlword & ~0x0f00) | 0x0200;   \
        if (new_387controlword != old_387controlword) {                 \
            _Py_set_387controlword(new_387controlword);                 \
        }                                                               \
    } while (0)
#define _Py_SET_53BIT_PRECISION_END                                     \
    do {                                                                \
        if (new_387controlword != old_387controlword) {                 \
            _Py_set_387controlword(old_387controlword);                 \
        }                                                               \
    } while (0)
#endif

// Get and set x87 control word for VisualStudio/x86.
// x87 is not supported in 64-bit or ARM.
#if defined(_MSC_VER) && !defined(_WIN64) && !defined(_M_ARM)
#define HAVE_PY_SET_53BIT_PRECISION 1

#include <float.h>                // __control87_2()

#define _Py_SET_53BIT_PRECISION_HEADER \
    unsigned int old_387controlword, new_387controlword, out_387controlword
    // We use the __control87_2 function to set only the x87 control word.
    // The SSE control word is unaffected.
#define _Py_SET_53BIT_PRECISION_START                                   \
    do {                                                                \
        __control87_2(0, 0, &old_387controlword, NULL);                 \
        new_387controlword =                                            \
          (old_387controlword & ~(_MCW_PC | _MCW_RC)) | (_PC_53 | _RC_NEAR); \
        if (new_387controlword != old_387controlword) {                 \
            __control87_2(new_387controlword, _MCW_PC | _MCW_RC,        \
                          &out_387controlword, NULL);                   \
        }                                                               \
    } while (0)
#define _Py_SET_53BIT_PRECISION_END                                     \
    do {                                                                \
        if (new_387controlword != old_387controlword) {                 \
            __control87_2(old_387controlword, _MCW_PC | _MCW_RC,        \
                          &out_387controlword, NULL);                   \
        }                                                               \
    } while (0)
#endif


// MC68881
#ifdef HAVE_GCC_ASM_FOR_MC68881
#define HAVE_PY_SET_53BIT_PRECISION 1
#define _Py_SET_53BIT_PRECISION_HEADER \
    unsigned int old_fpcr, new_fpcr
#define _Py_SET_53BIT_PRECISION_START                                   \
    do {                                                                \
        __asm__ ("fmove.l %%fpcr,%0" : "=g" (old_fpcr));                \
        /* Set double precision / round to nearest.  */                 \
        new_fpcr = (old_fpcr & ~0xf0) | 0x80;                           \
        if (new_fpcr != old_fpcr) {                                     \
              __asm__ volatile ("fmove.l %0,%%fpcr" : : "g" (new_fpcr));\
        }                                                               \
    } while (0)
#define _Py_SET_53BIT_PRECISION_END                                     \
    do {                                                                \
        if (new_fpcr != old_fpcr) {                                     \
            __asm__ volatile ("fmove.l %0,%%fpcr" : : "g" (old_fpcr));  \
        }                                                               \
    } while (0)
#endif

// Default definitions are empty
#ifndef _Py_SET_53BIT_PRECISION_HEADER
#  define _Py_SET_53BIT_PRECISION_HEADER
#  define _Py_SET_53BIT_PRECISION_START
#  define _Py_SET_53BIT_PRECISION_END
#endif


//--- _PY_SHORT_FLOAT_REPR macro -------------------------------------------

// If we can't guarantee 53-bit precision, don't use the code
// in Python/dtoa.c, but fall back to standard code.  This
// means that repr of a float will be long (17 significant digits).
//
// Realistically, there are two things that could go wrong:
//
// (1) doubles aren't IEEE 754 doubles, or
// (2) we're on x86 with the rounding precision set to 64-bits
//     (extended precision), and we don't know how to change
//     the rounding precision.
#if !defined(DOUBLE_IS_LITTLE_ENDIAN_IEEE754) && \
    !defined(DOUBLE_IS_BIG_ENDIAN_IEEE754) && \
    !defined(DOUBLE_IS_ARM_MIXED_ENDIAN_IEEE754)
#  define _PY_SHORT_FLOAT_REPR 0
#endif

// Double rounding is symptomatic of use of extended precision on x86.
// If we're seeing double rounding, and we don't have any mechanism available
// for changing the FPU rounding precision, then don't use Python/dtoa.c.
#if defined(X87_DOUBLE_ROUNDING) && !defined(HAVE_PY_SET_53BIT_PRECISION)
#  define _PY_SHORT_FLOAT_REPR 0
#endif

#ifndef _PY_SHORT_FLOAT_REPR
#  define _PY_SHORT_FLOAT_REPR 1
#endif


#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_PYMATH_H */
