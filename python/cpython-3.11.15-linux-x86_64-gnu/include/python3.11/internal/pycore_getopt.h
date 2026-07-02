#ifndef Py_INTERNAL_PYGETOPT_H
#define Py_INTERNAL_PYGETOPT_H

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

extern int _PyOS_opterr;
extern Py_ssize_t _PyOS_optind;
extern const wchar_t *_PyOS_optarg;

extern void _PyOS_ResetGetOpt(void);

typedef struct {
    const wchar_t *name;
    int has_arg;
    int val;
} _PyOS_LongOption;

extern int _PyOS_GetOpt(Py_ssize_t argc, wchar_t * const *argv, int *longindex);

#endif /* !Py_INTERNAL_PYGETOPT_H */
